from __future__ import annotations

from cw360.teacher.candidate import TeacherCandidate
from cw360.teacher.engine import TeacherEngineConfig, TeacherGenerationEngine
from cw360.teacher.providers.mock import MockTeacherClient
from cw360.teacher.queue import build_request
from cw360.teacher.response_validation import validate_response_payload


def test_cached_payload_is_renormalized_even_if_it_claims_engine_normalized(tmp_path) -> None:
    candidate = TeacherCandidate(
        candidate_id="real-candidate-1",
        source="debugbench",
        task_type="debugging",
        score=0.94,
        input_hash="real-input-hash",
        estimated_tokens=96,
        reason_selected="strong_debug_signal",
        quality_tier="A",
        proposed_train_stage="instruction_tune",
        proposed_loss_weight=1.0,
        raw_text="def divide(a, b):\n    return a / b\n",
    )
    request = build_request(candidate)
    cached_payload = {
        "difficulty": "beginner",
        "skills": ["python", "debugging", "edge_cases"],
        "user_prompt": "Debug this Python divide function for edge cases.",
        "assistant_response": (
            "The Python bug is that division by zero is not handled. Add a guard for "
            "b == 0, raise ValueError, and cover the edge case with pytest."
        ),
        "teacher_provider": "manual_probe",
        "teacher_model": "fake_model",
        "source_candidate_id": "fake_candidate",
        "source": "fake_source",
        "task_type": "fake_task",
        "train_stage": "fake_stage",
        "loss_weight": 999,
        "input_hash": "fake_input_hash",
        "output_hash": "fake_output_hash",
        "synthetic_id": "fake_synthetic_id",
        "metadata": {"normalized_by_engine": True},
    }
    engine = TeacherGenerationEngine(
        TeacherEngineConfig(
            run_id="run-1",
            provider="mock",
            model="real-model",
            storage={"local_temp_dir": str(tmp_path), "shard_size": 10},
        ),
        client=MockTeacherClient(model_name="real-model"),
    )

    normalized = engine._normalize_cached_payload(cached_payload, request)

    assert normalized["teacher_provider"] == "mock"
    assert normalized["teacher_model"] == "real-model"
    assert normalized["source_candidate_id"] == "real-candidate-1"
    assert normalized["source"] == "debugbench"
    assert normalized["task_type"] == "debugging"
    assert normalized["train_stage"] == "instruction_tune"
    assert normalized["loss_weight"] == 1.0
    assert normalized["input_hash"] == "real-input-hash"
    assert normalized["output_hash"] != "fake_output_hash"
    assert normalized["synthetic_id"] != "fake_synthetic_id"

    for fake_value in (
        "manual_probe",
        "fake_model",
        "fake_candidate",
        "fake_source",
        "fake_task",
        "fake_stage",
        "fake_input_hash",
        "fake_output_hash",
        "fake_synthetic_id",
    ):
        assert fake_value not in normalized.values()

    validation = validate_response_payload(normalized, source_prompt=request.prompt)
    assert validation.ok, validation.errors
