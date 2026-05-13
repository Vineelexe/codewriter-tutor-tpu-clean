from __future__ import annotations

import pytest

from cw360.synthetic.schemas import StructuredSyntheticRecord, SyntheticSchemaError


def sample_payload(**overrides):
    payload = {
        "synthetic_id": "synth-1",
        "source_candidate_id": "candidate-1",
        "candidate_id": "raw-candidate-1",
        "source": "debugbench",
        "record_type": "structured",
        "task_type": "debugging",
        "difficulty": "beginner",
        "skills": ["python", "debugging", "pytest"],
        "user_prompt": "Debug this Python function that divides by zero.",
        "assistant_response": (
            "The Python bug is a missing zero-count guard. Fix it before division:\n\n"
            "```python\n"
            "def average(total, count):\n"
            "    if count == 0:\n"
            "        return 0\n"
            "    return total / count\n"
            "```"
        ),
        "train_stage": "instruction_tune",
        "loss_weight": 1.0,
        "teacher_provider": "mock",
        "teacher_model": "mock-teacher",
        "generation_time": "2026-05-11T00:00:00Z",
        "input_hash": "input-hash",
        "output_hash": "output-hash",
        "metadata": {"fixture": True},
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


def test_structured_synthetic_record_round_trips_optional_fields() -> None:
    record = StructuredSyntheticRecord.from_dict(sample_payload())

    payload = record.to_dict()
    reloaded = StructuredSyntheticRecord.from_dict(payload)

    assert reloaded.to_dict() == payload
    assert reloaded.candidate_id == "raw-candidate-1"
    assert reloaded.stable_candidate_key == "candidate-1"


def test_structured_synthetic_record_reports_missing_required_field() -> None:
    payload = sample_payload()
    payload.pop("synthetic_id")

    with pytest.raises(SyntheticSchemaError, match="synthetic_id"):
        StructuredSyntheticRecord.from_dict(payload)
