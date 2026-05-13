from __future__ import annotations

import json
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from cw360.checkpoint import metadata_path_for_checkpoint
from cw360.config import ModelConfig, load_model_config
from cw360.inference.generate import generate_text
from cw360.model.count import (
    ParameterCountReport,
    build_parameter_report,
    validate_size_label_count,
)
from cw360.model.lm import CodeWriterTutorLM
from cw360.prepack.config import load_prepack_config
from cw360.synthetic.validate import validate_record_payload
from cw360.teacher.candidate import TeacherCandidate
from cw360.teacher.prompts import build_teacher_prompt
from cw360.teacher.providers.mock import MockTeacherClient
from cw360.teacher.response_validation import validate_json_response
from cw360.tokenizer.loader import ensure_pad_token_for_batching, load_tokenizer
from cw360.train.cpu_trainer import CPUTinyTrainer

TokenizerLoader = Callable[[str], Any]


class LocalRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class LocalRuntimeReport:
    name: str
    ok: bool
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assert_tiny_cpu_forward_allowed(config: ModelConfig) -> None:
    if config.size_label != "tiny":
        raise LocalRuntimeError(
            "local forward/training is allowed only for tiny configs; use --no-forward for "
            f"{config.size_label} parameter counts and memory estimates"
        )


def run_full_config_count_check(
    config_path: str | Path,
    *,
    no_forward: bool,
) -> ParameterCountReport:
    config = load_model_config(config_path)
    if not no_forward:
        raise LocalRuntimeError(
            "full candidate configs must use --no-forward; local Phase 12.5 never "
            "instantiates 354M/420M/480M models"
        )
    return build_parameter_report(config)


def run_tiny_model_step_check(
    config_path: str | Path,
    *,
    tokenizer_loader: TokenizerLoader | None = None,
) -> LocalRuntimeReport:
    config = load_model_config(config_path)
    assert_tiny_cpu_forward_allowed(config)
    tokenizer = _load_runtime_tokenizer(config, tokenizer_loader)
    model = CodeWriterTutorLM(config)
    model.train()
    input_ids = torch.randint(0, config.vocab_size, (1, 4), dtype=torch.long)
    output = model(input_ids, labels=input_ids)
    if output.loss is None:
        raise LocalRuntimeError("tiny forward did not return a loss")
    output.loss.backward()
    gradient_count = sum(
        1
        for parameter in model.parameters()
        if parameter.grad is not None and parameter.grad.abs().sum() > 0
    )
    if gradient_count <= 0:
        raise LocalRuntimeError("tiny backward produced no nonzero gradients")
    return LocalRuntimeReport(
        name="tiny_model_step",
        ok=True,
        details={
            "device": "cpu",
            "tokenizer_name": config.tokenizer_name,
            "tokenizer_class": type(tokenizer).__name__,
            "logits_shape": list(output.logits.shape),
            "loss": float(output.loss.detach().item()),
            "gradient_tensors": gradient_count,
        },
    )


def run_tiny_inference_check(
    config_path: str | Path,
    *,
    prompt: str,
    checkpoint_path: str | Path | None = None,
    max_new_tokens: int = 8,
    tokenizer_loader: TokenizerLoader | None = None,
) -> LocalRuntimeReport:
    config = load_model_config(config_path)
    assert_tiny_cpu_forward_allowed(config)
    tokenizer = _load_runtime_tokenizer(config, tokenizer_loader)
    model = CodeWriterTutorLM(config)
    if checkpoint_path is not None:
        from cw360.checkpoint import load_checkpoint

        load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            expected_model_size_label="tiny",
            map_location="cpu",
        )
    model.eval()
    text = generate_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        seed=1234,
        device="cpu",
    )
    return LocalRuntimeReport(
        name="tiny_inference",
        ok=True,
        details={"prompt": prompt, "generated_text": text, "device": "cpu"},
    )


def run_tiny_synthetic_train_resume_check(
    *,
    model_config_path: str | Path,
    output_dir: str | Path,
    steps: int,
    first_steps: int | None = None,
    seq_len: int = 16,
    batch_size: int = 1,
) -> LocalRuntimeReport:
    config = _tiny_train_config(
        model_config_path=model_config_path,
        max_steps=max(1, first_steps if first_steps is not None else steps // 2),
        seq_len=seq_len,
        batch_size=batch_size,
    )
    output = Path(output_dir)
    first = CPUTinyTrainer(config=config, output_dir=output)
    first_result = first.train()
    resumed_config = dict(config)
    resumed_config["max_steps"] = int(steps)
    resumed = CPUTinyTrainer(
        config=resumed_config,
        output_dir=output,
        resume_checkpoint=first_result.checkpoints[-1],
    )
    second_result = resumed.train()
    final_checkpoint = second_result.checkpoints[-1]
    metadata_path = metadata_path_for_checkpoint(final_checkpoint)
    return LocalRuntimeReport(
        name="tiny_synthetic_train_resume",
        ok=True,
        details={
            "final_step": second_result.state.step,
            "tokens_seen": second_result.state.tokens_seen,
            "sequences_seen": second_result.state.sequences_seen,
            "resumed_from": str(first_result.checkpoints[-1]),
            "final_checkpoint": str(final_checkpoint),
            "metadata_path": str(metadata_path),
            "metadata_exists": metadata_path.exists(),
        },
    )


def run_tiny_prepacked_train_resume_check(
    *,
    model_config_path: str | Path,
    output_dir: str | Path,
    prepacked_dir: str | Path,
    steps: int,
    first_steps: int | None = None,
    seq_len: int = 16,
    batch_size: int = 1,
    num_sequences: int | None = None,
) -> LocalRuntimeReport:
    from cw360.train.shard_dataloader import (
        PrepackedShardDataLoader,
        data_snapshot_from_prepacked_manifest,
        validate_prepacked_manifest_for_training,
    )

    prepacked_root = Path(prepacked_dir)
    if not (prepacked_root / "manifest.json").exists():
        from scripts.make_tiny_prepacked_dataset import make_tiny_prepacked_dataset

        config = load_prepack_config("configs/prepack_tpu_1024.yaml")
        make_tiny_prepacked_dataset(
            config=config,
            output_dir=prepacked_root,
            num_sequences=num_sequences or max(steps + 2, 4),
        )
    train_config = _tiny_train_config(
        model_config_path=model_config_path,
        max_steps=max(1, first_steps if first_steps is not None else steps // 2),
        seq_len=seq_len,
        batch_size=batch_size,
    )
    train_config["max_seq_len"] = 1024
    train_config["local_cpu_train_seq_len"] = seq_len
    train_config["prepacked_shards_path"] = str(prepacked_root)
    model_config = load_model_config(model_config_path)
    snapshot = data_snapshot_from_prepacked_manifest(prepacked_root)
    manifest = PrepackedShardDataLoader(
        prepacked_root,
        split="train",
        batch_size=batch_size,
        drop_last=True,
        static_shapes=True,
        shard_order_seed=0,
    ).manifest
    validate_prepacked_manifest_for_training(manifest)

    def train_factory(cursor):
        return _modulo_batches(
            iter(
                PrepackedShardDataLoader(
                    prepacked_root,
                    split="train",
                    batch_size=batch_size,
                    drop_last=True,
                    static_shapes=True,
                    shard_order_seed=0,
                    cursor=cursor,
                )
            ),
            vocab_size=model_config.vocab_size,
        )

    def val_factory(cursor):
        del cursor
        return _modulo_batches(
            iter(
                PrepackedShardDataLoader(
                    prepacked_root,
                    split="val",
                    batch_size=batch_size,
                    drop_last=True,
                    static_shapes=True,
                    shard_order_seed=0,
                )
            ),
            vocab_size=model_config.vocab_size,
        )

    output = Path(output_dir)
    first = CPUTinyTrainer(
        config=train_config,
        output_dir=output,
        batch_factory=train_factory,
        val_batch_factory=val_factory,
        data_snapshot=snapshot,
    )
    first_result = first.train()
    resumed_config = dict(train_config)
    resumed_config["max_steps"] = int(steps)
    resumed = CPUTinyTrainer(
        config=resumed_config,
        output_dir=output,
        batch_factory=train_factory,
        val_batch_factory=val_factory,
        data_snapshot=snapshot,
        resume_checkpoint=first_result.checkpoints[-1],
    )
    second_result = resumed.train()
    cursor = resumed.data_cursor
    if cursor is None:
        raise LocalRuntimeError("prepacked training did not produce a data cursor")
    return LocalRuntimeReport(
        name="tiny_prepacked_train_resume",
        ok=True,
        details={
            "final_step": second_result.state.step,
            "tokens_seen": second_result.state.tokens_seen,
            "sequences_seen": second_result.state.sequences_seen,
            "resumed_from": str(first_result.checkpoints[-1]),
            "final_checkpoint": str(second_result.checkpoints[-1]),
            "prepacked_dir": str(prepacked_root),
            "cursor": cursor.to_dict(),
        },
    )


def run_teacher_mock_check() -> LocalRuntimeReport:
    client = MockTeacherClient()
    bundle = build_teacher_prompt(_teacher_candidate())
    response = client.generate(bundle.prompt, bundle.system_prompt, 0.2, 1200)
    validation = validate_json_response(response.text, source_prompt=bundle.prompt)
    if not validation.ok or validation.record is None:
        raise LocalRuntimeError(f"teacher mock validation failed: {validation.errors}")
    return LocalRuntimeReport(
        name="teacher_mock",
        ok=True,
        details={
            "provider": validation.record.teacher_provider,
            "task_type": validation.record.task_type,
            "calls": client.calls,
        },
    )


def run_synthetic_validation_check() -> LocalRuntimeReport:
    accepted = validate_record_payload(_synthetic_payload())
    rejected = validate_record_payload(
        _synthetic_payload(
            task_type="general_chat",
            user_prompt="Tell me anything.",
            assistant_response="Anything.",
            source_candidate_id="",
            candidate_id=None,
        )
    )
    if not accepted.valid:
        raise LocalRuntimeError(f"valid Python synthetic record was rejected: {accepted.reasons}")
    if rejected.valid or "invalid_task_type" not in rejected.reasons:
        raise LocalRuntimeError("general-chat synthetic drift was not rejected")
    return LocalRuntimeReport(
        name="synthetic_validation",
        ok=True,
        details={
            "accepted_task": accepted.record.task_type if accepted.record else None,
            "rejected": rejected.reasons,
        },
    )


def run_local_smoke(
    *,
    config_path: str | Path,
    prompt: str,
    steps: int = 20,
    tokenizer_loader: TokenizerLoader | None = None,
) -> list[LocalRuntimeReport]:
    with tempfile.TemporaryDirectory(prefix="cw360-local-smoke-") as tmp:
        root = Path(tmp)
        return [
            run_tiny_model_step_check(config_path, tokenizer_loader=tokenizer_loader),
            run_tiny_synthetic_train_resume_check(
                model_config_path=config_path,
                output_dir=root / "synthetic_ckpts",
                steps=steps,
            ),
            run_tiny_prepacked_train_resume_check(
                model_config_path=config_path,
                output_dir=root / "prepacked_ckpts",
                prepacked_dir=root / "prepacked",
                steps=steps,
            ),
            run_teacher_mock_check(),
            run_synthetic_validation_check(),
            run_tiny_inference_check(
                config_path,
                prompt=prompt,
                tokenizer_loader=tokenizer_loader,
            ),
        ]


def reports_to_json(reports: list[LocalRuntimeReport]) -> str:
    return json.dumps([report.to_dict() for report in reports], indent=2, sort_keys=True)


def _load_runtime_tokenizer(
    config: ModelConfig,
    tokenizer_loader: TokenizerLoader | None,
) -> Any:
    try:
        tokenizer = (
            tokenizer_loader(config.tokenizer_name)
            if tokenizer_loader is not None
            else load_tokenizer(config.tokenizer_name)
        )
        ensure_pad_token_for_batching(tokenizer)
        return tokenizer
    except Exception as exc:
        raise LocalRuntimeError(
            "failed to load StarCoder2 tokenizer for local runtime. Populate the Hugging Face "
            "cache or set HF_TOKEN if needed; unit tests must mock tokenizer loading."
        ) from exc


def _tiny_train_config(
    *,
    model_config_path: str | Path,
    max_steps: int,
    seq_len: int,
    batch_size: int,
) -> dict[str, Any]:
    return {
        "config_type": "train",
        "training_stage": "base_pretrain",
        "platform": "local-cpu-test",
        "model_config_path": str(model_config_path),
        "max_seq_len": int(seq_len),
        "micro_batch_size": int(batch_size),
        "gradient_accumulation_steps": 1,
        "learning_rate": 3.0e-4,
        "weight_decay": 0.0,
        "max_steps": int(max_steps),
        "warmup_steps": 0,
        "max_grad_norm": 1.0,
        "local_cpu_train_seq_len": int(seq_len),
        "drop_last": True,
        "static_shapes": True,
        "notes": "phase12.5 local runtime check",
    }


def _modulo_batches(iterator, *, vocab_size: int):
    from cw360.train.trainer import TrainingBatch

    for batch in iterator:
        yield TrainingBatch(
            input_ids=batch.input_ids.remainder(vocab_size),
            labels=batch.labels.remainder(vocab_size),
            tpu_data_cursor=batch.tpu_data_cursor,
            data_wait_time=batch.data_wait_time,
        )


def write_tiny_unit_model_config(path: str | Path) -> Path:
    config_path = Path(path)
    payload = {
        "config_type": "model",
        "model_name": "cw360",
        "size_label": "tiny",
        "tokenizer_name": "bigcode/starcoder2-15b",
        "vocab_size": 32,
        "max_position_embeddings": 32,
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 32,
        "activation": "silu",
        "norm_eps": 1.0e-5,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": True,
        "use_bias": False,
        "attention_dropout": 0.0,
        "residual_dropout": 0.0,
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def _teacher_candidate() -> TeacherCandidate:
    return TeacherCandidate(
        candidate_id="local-runtime-candidate",
        source="local-runtime",
        task_type="debugging",
        score=0.9,
        input_hash="local-input",
        estimated_tokens=64,
        reason_selected="phase12.5 mock check",
        quality_tier="A",
        proposed_train_stage="instruction_tune",
        proposed_loss_weight=1.0,
        raw_text="def divide(a, b):\n    return a / b\n",
    )


def _synthetic_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "synthetic_id": "local-synth-1",
        "source_candidate_id": "local-candidate-1",
        "candidate_id": "raw-local-candidate-1",
        "source": "local-runtime",
        "record_type": "structured",
        "task_type": "debugging",
        "difficulty": "beginner",
        "skills": ["python", "debugging", "pytest"],
        "user_prompt": "Debug this Python function that divides by zero.",
        "assistant_response": (
            "The Python bug is a missing zero-count guard. Add a guard before division."
        ),
        "train_stage": "instruction_tune",
        "loss_weight": 1.0,
        "teacher_provider": "mock",
        "teacher_model": "mock-teacher",
        "generation_time": "2026-05-11T00:00:00Z",
        "input_hash": "input-hash",
        "output_hash": "output-hash",
        "metadata": {"phase": "12.5"},
        "code_before": "def average(total, count):\n    return total / count\n",
        "code_after": (
            "def average(total, count):\n"
            "    if count == 0:\n"
            "        return 0\n"
            "    return total / count\n"
        ),
        "tests": "def test_zero_count():\n    assert average(1, 0) == 0\n",
    }
    payload.update(overrides)
    return payload


def parameter_report_to_dict(report: ParameterCountReport) -> dict[str, Any]:
    return {
        "size_label": report.size_label,
        "total_parameters": report.total_parameters,
        "trainable_parameters": report.trainable_parameters,
        "embedding_parameters": report.embedding_parameters,
        "per_block_estimate": report.per_block_estimate,
        "expected_parameters": report.expected_parameters,
        "actual_parameters": report.actual_parameters,
        "delta_from_expected": report.delta_from_expected,
        "relative_delta_from_expected": report.relative_delta_from_expected,
        "size_label_valid": report.size_label_valid,
        "within_label_tolerance": validate_size_label_count(report),
    }


def train_config_from_mapping(config: Mapping[str, Any]) -> dict[str, Any]:
    return dict(config)
