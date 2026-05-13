from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import torch

from cw360.checkpoint.manager import load_checkpoint
from cw360.config import ModelConfig, load_model_config, load_yaml
from cw360.eval.export import write_jsonl, write_markdown_report
from cw360.eval.prompts import EvalPrompt, load_eval_prompts
from cw360.eval.scoring_schema import scoring_schema_as_dict
from cw360.model.lm import CodeWriterTutorLM
from cw360.tokenizer.loader import ensure_pad_token_for_batching, load_tokenizer


class TokenizerLike(Protocol):
    eos_token_id: int | None

    def __call__(self, text: str, **kwargs: Any) -> Any: ...

    def decode(self, token_ids: Any, **kwargs: Any) -> str: ...


@dataclass(frozen=True, slots=True)
class EvalRunConfig:
    model_config_path: Path
    prompt_paths: tuple[Path, ...]
    output_dir: Path
    checkpoint_path: Path | None = None
    checkpoint_name: str | None = None
    step: int = 0
    temperature: float = 0.0
    max_new_tokens: int = 128
    notes: str = "fixed evaluation prompts"
    training_stage: str = "eval_only"
    model_size_label: str | None = None
    data_snapshot: Any = "not_trained_random_baseline"
    top_k: int | None = None
    top_p: float | None = None
    init_seed: int = 1234
    use_cache: bool = True

    @classmethod
    def from_eval_config(
        cls,
        eval_config_path: str | Path,
        *,
        model_config_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_name: str | None = None,
        output_dir: str | Path | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        random_baseline: bool = False,
    ) -> EvalRunConfig:
        eval_config = load_yaml(eval_config_path)
        resolved_model_config = Path(
            model_config_path if model_config_path is not None else eval_config["model_config_path"]
        )
        prompt_values = eval_config.get("prompt_paths")
        if prompt_values is None:
            prompt_values = [
                eval_config.get("core_prompt_path", "eval_prompts/core_prompts.jsonl"),
                eval_config.get("extended_prompt_path", "eval_prompts/extended_prompts.jsonl"),
            ]
        prompt_paths = tuple(Path(str(path)) for path in prompt_values)
        resolved_checkpoint = None if checkpoint_path is None else Path(checkpoint_path)
        resolved_checkpoint_name = checkpoint_name or (
            "random_init" if random_baseline else _checkpoint_name_from_path(resolved_checkpoint)
        )
        return cls(
            model_config_path=resolved_model_config,
            prompt_paths=prompt_paths,
            output_dir=Path(
                output_dir
                if output_dir is not None
                else eval_config.get("output_dir", "outputs/eval")
            ),
            checkpoint_path=resolved_checkpoint,
            checkpoint_name=resolved_checkpoint_name,
            temperature=float(
                temperature if temperature is not None else eval_config.get("temperature", 0.0)
            ),
            max_new_tokens=int(
                max_new_tokens
                if max_new_tokens is not None
                else eval_config.get("max_new_tokens", 128)
            ),
            notes=str(eval_config.get("notes", "fixed evaluation prompts")),
            training_stage=str(eval_config.get("training_stage", "eval_only")),
            model_size_label=(
                None
                if eval_config.get("model_size_label") is None
                else str(eval_config["model_size_label"])
            ),
            data_snapshot=eval_config.get("data_snapshot", "not_trained_random_baseline"),
            top_k=None if eval_config.get("top_k") is None else int(eval_config["top_k"]),
            top_p=None if eval_config.get("top_p") is None else float(eval_config["top_p"]),
            init_seed=int(eval_config.get("init_seed", 1234)),
            use_cache=bool(eval_config.get("use_cache", True)),
        )


def run_evaluation(
    config: EvalRunConfig,
    *,
    model: CodeWriterTutorLM | None = None,
    tokenizer: TokenizerLike | None = None,
    device: str | torch.device = "cpu",
) -> tuple[Path, Path]:
    model_config = load_model_config(config.model_config_path)
    resolved_model = (
        model if model is not None else CodeWriterTutorLM(model_config, init_seed=config.init_seed)
    )
    resolved_model.to(device)
    resolved_model.eval()

    checkpoint_name = config.checkpoint_name or _checkpoint_name_from_path(config.checkpoint_path)
    step = config.step
    training_stage = config.training_stage
    data_snapshot = config.data_snapshot
    notes = config.notes
    if config.checkpoint_path is not None:
        metadata, _ = load_checkpoint(
            checkpoint_path=config.checkpoint_path,
            model=resolved_model,
            expected_model_size_label=model_config.size_label,
            allow_training_stage_mismatch=True,
            allow_prepacked_dataset_mismatch=True,
            map_location=device,
        )
        checkpoint_name = config.checkpoint_name or _checkpoint_name_from_path(
            config.checkpoint_path
        )
        step = metadata.step
        training_stage = metadata.training_stage
        data_snapshot = metadata.data_snapshot.to_dict()
        notes = metadata.notes or config.notes

    resolved_tokenizer = (
        tokenizer if tokenizer is not None else load_tokenizer(model_config.tokenizer_name)
    )
    if hasattr(resolved_tokenizer, "pad_token"):
        ensure_pad_token_for_batching(resolved_tokenizer)  # type: ignore[arg-type]

    prompts = load_eval_prompts(config.prompt_paths)
    date = datetime.now(timezone.utc).isoformat(timespec="seconds")
    records = [
        _run_single_prompt(
            prompt,
            model=resolved_model,
            tokenizer=resolved_tokenizer,
            model_config=model_config,
            device=device,
            checkpoint_name=checkpoint_name,
            step=step,
            temperature=config.temperature,
            max_new_tokens=config.max_new_tokens,
            date=date,
            notes=notes,
            training_stage=training_stage,
            model_size_label=config.model_size_label or model_config.size_label,
            data_snapshot=data_snapshot,
            top_k=config.top_k,
            top_p=config.top_p,
            use_cache=config.use_cache,
        )
        for prompt in prompts
    ]
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "results.jsonl"
    markdown_path = output_dir / "report.md"
    write_jsonl(records, jsonl_path)
    write_markdown_report(records, markdown_path)
    return jsonl_path, markdown_path


def _run_single_prompt(
    prompt: EvalPrompt,
    *,
    model: CodeWriterTutorLM,
    tokenizer: TokenizerLike,
    model_config: ModelConfig,
    device: str | torch.device,
    checkpoint_name: str,
    step: int,
    temperature: float,
    max_new_tokens: int,
    date: str,
    notes: str,
    training_stage: str,
    model_size_label: str,
    data_snapshot: Any,
    top_k: int | None,
    top_p: float | None,
    use_cache: bool,
) -> dict[str, Any]:
    encoded = tokenizer(
        prompt.prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max(1, model_config.max_position_embeddings - max_new_tokens),
    )
    input_ids = _extract_tensor(encoded, "input_ids").to(device)
    attention_mask = _extract_optional_tensor(encoded, "attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            attention_mask=attention_mask if use_cache else None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            use_cache=use_cache,
        )
    new_tokens = generated[:, input_ids.shape[1] :][0].detach().cpu()
    output = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return {
        "prompt_id": prompt.prompt_id,
        "category": prompt.category,
        "prompt": prompt.prompt,
        "output": output,
        "checkpoint_name": checkpoint_name,
        "step": step,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "date": date,
        "notes": notes,
        "training_stage": training_stage,
        "model_size_label": model_size_label,
        "data_snapshot": data_snapshot,
        "scoring_schema": scoring_schema_as_dict(),
    }


def _extract_tensor(encoded: Any, key: str) -> torch.Tensor:
    if isinstance(encoded, dict):
        value = encoded[key]
    else:
        value = getattr(encoded, key)
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, dtype=torch.long)
    return value


def _extract_optional_tensor(encoded: Any, key: str) -> torch.Tensor | None:
    if isinstance(encoded, dict):
        value = encoded.get(key)
    else:
        value = getattr(encoded, key, None)
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, dtype=torch.long)
    return value


def _checkpoint_name_from_path(path: Path | None) -> str:
    if path is None:
        return "random_init"
    return path.stem
