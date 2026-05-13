from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cw360.config import load_model_config
from cw360.local.runtime import (
    LocalRuntimeError,
    assert_tiny_cpu_forward_allowed,
    parameter_report_to_dict,
    run_full_config_count_check,
    run_synthetic_validation_check,
    run_teacher_mock_check,
    run_tiny_inference_check,
    run_tiny_model_step_check,
    run_tiny_prepacked_train_resume_check,
    run_tiny_synthetic_train_resume_check,
    write_tiny_unit_model_config,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


class FakeTokenizer:
    eos_token = "<eos>"
    eos_token_id = 2
    pad_token = None

    def __call__(self, prompt: str, return_tensors: str):
        del prompt, return_tensors
        return {
            "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(f"tok{int(token)}" for token in token_ids)


def test_tiny_model_step_loads_mock_tokenizer_and_flows_gradients(tmp_path: Path) -> None:
    config_path = write_tiny_unit_model_config(tmp_path / "model_tiny_unit.yaml")

    report = run_tiny_model_step_check(config_path, tokenizer_loader=lambda _: FakeTokenizer())

    assert report.ok
    assert report.details["device"] == "cpu"
    assert report.details["gradient_tensors"] > 0
    assert report.details["logits_shape"] == [1, 4, 32]


def test_tiny_inference_uses_fake_tokenizer(tmp_path: Path) -> None:
    config_path = write_tiny_unit_model_config(tmp_path / "model_tiny_unit.yaml")

    report = run_tiny_inference_check(
        config_path,
        prompt="Write a tiny Python function.",
        tokenizer_loader=lambda _: FakeTokenizer(),
        max_new_tokens=2,
    )

    assert report.ok
    assert report.details["device"] == "cpu"
    assert isinstance(report.details["generated_text"], str)


def test_tiny_synthetic_train_resume_is_fast_and_writes_metadata(tmp_path: Path) -> None:
    config_path = write_tiny_unit_model_config(tmp_path / "model_tiny_unit.yaml")

    report = run_tiny_synthetic_train_resume_check(
        model_config_path=config_path,
        output_dir=tmp_path / "synthetic_ckpts",
        steps=2,
        first_steps=1,
        seq_len=8,
        batch_size=1,
    )

    assert report.details["final_step"] == 2
    assert report.details["metadata_exists"]


def test_tiny_prepacked_train_resume_is_fast_and_has_cursor(tmp_path: Path) -> None:
    config_path = write_tiny_unit_model_config(tmp_path / "model_tiny_unit.yaml")

    report = run_tiny_prepacked_train_resume_check(
        model_config_path=config_path,
        output_dir=tmp_path / "prepacked_ckpts",
        prepacked_dir=tmp_path / "prepacked",
        steps=2,
        first_steps=1,
        seq_len=8,
        batch_size=1,
        num_sequences=4,
    )

    cursor = report.details["cursor"]
    assert report.details["final_step"] == 2
    assert cursor["epoch"] == 0
    assert cursor["shard_order_seed"] == 0
    assert cursor["shard_index"] >= 0
    assert cursor["sequence_offset"] >= 0
    assert cursor["tokens_seen"] > 0
    assert cursor["sequences_seen"] > 0


def test_teacher_mock_and_synthetic_validation_stay_python_scoped() -> None:
    teacher = run_teacher_mock_check()
    synthetic = run_synthetic_validation_check()

    assert teacher.details["provider"] == "mock"
    assert teacher.details["task_type"] == "debugging"
    assert synthetic.details["accepted_task"] == "debugging"
    assert "invalid_task_type" in synthetic.details["rejected"]


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("model_354m.yaml", 354_000_000),
        ("model_420m.yaml", 423_000_000),
        ("model_480m.yaml", 479_000_000),
    ],
)
def test_full_candidate_configs_are_no_forward_only(filename: str, expected: int) -> None:
    config_path = CONFIG_DIR / filename

    report = run_full_config_count_check(config_path, no_forward=True)
    payload = parameter_report_to_dict(report)

    assert payload["expected_parameters"] == expected
    assert payload["within_label_tolerance"]
    assert payload["actual_parameters"] is None
    with pytest.raises(LocalRuntimeError, match="must use --no-forward"):
        run_full_config_count_check(config_path, no_forward=False)
    with pytest.raises(LocalRuntimeError, match="only for tiny configs"):
        assert_tiny_cpu_forward_allowed(load_model_config(config_path))
