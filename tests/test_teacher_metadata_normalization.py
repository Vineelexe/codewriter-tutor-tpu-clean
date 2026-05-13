from __future__ import annotations

from cw360.teacher.candidate import TeacherCandidate
from cw360.teacher.normalization import normalize_teacher_payload
from cw360.teacher.queue import build_request
from cw360.teacher.response_validation import validate_response_payload


def test_teacher_metadata_poisoning_is_overwritten_before_validation() -> None:
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
    raw_payload = {
        "difficulty": "beginner",
        "skills": ["python", "debugging", "edge_cases"],
        "user_prompt": "Debug this Python divide function for edge cases.",
        "assistant_response": (
            "The Python bug is that division by zero is not handled. Add a guard for "
            "b == 0, raise ValueError, and cover the edge case with pytest."
        ),
        "code_before": "def divide(a, b):\n    return a / b\n",
        "code_after": (
            "def divide(a, b):\n"
            "    if b == 0:\n"
            "        raise ValueError('b must not be zero')\n"
            "    return a / b\n"
        ),
        "explanation": "The guard makes the Python error explicit before division.",
        "tests": "def test_divide_zero():\n    with pytest.raises(ValueError): divide(1, 0)\n",
        "quality_notes": "Specific Python debugging example.",
        "teacher_provider": "manual_probe",
        "teacher_model": "fake_model",
        "output_hash": "fake_hash",
        "input_hash": "fake_input_hash",
        "source_candidate_id": "fake_candidate",
        "source": "fake_source",
        "task_type": "fake_task",
        "train_stage": "eval_only",
        "loss_weight": 999,
        "synthetic_id": "fake_id",
    }

    normalized = normalize_teacher_payload(
        raw_payload,
        request=request,
        provider_name="openai_compatible",
        model_name="llama-3.1-8b-instant",
        response_request_id="real-response-id",
        response_raw={"usage": {"total_tokens": 123}},
    )

    assert normalized["teacher_provider"] == "openai_compatible"
    assert normalized["teacher_model"] == "llama-3.1-8b-instant"
    assert normalized["source_candidate_id"] == "real-candidate-1"
    assert normalized["source"] == "debugbench"
    assert normalized["task_type"] == "debugging"
    assert normalized["train_stage"] == "instruction_tune"
    assert normalized["loss_weight"] == 1.0
    assert normalized["input_hash"] == "real-input-hash"
    assert normalized["output_hash"] != "fake_hash"
    assert normalized["synthetic_id"] != "fake_id"
    assert normalized["metadata"]["normalized_by_engine"] is True
    assert normalized["metadata"]["candidate_score"] == 0.94
    assert normalized["metadata"]["quality_tier"] == "A"
    assert normalized["metadata"]["reason_selected"] == "strong_debug_signal"
    assert normalized["metadata"]["request_hash"] == request.request_hash
    assert normalized["metadata"]["teacher_response_request_id"] == "real-response-id"

    blocked = set(normalized["metadata"]["llm_supplied_blocked_fields"])
    for field in (
        "teacher_provider",
        "teacher_model",
        "output_hash",
        "input_hash",
        "source_candidate_id",
        "source",
        "task_type",
        "train_stage",
        "loss_weight",
        "synthetic_id",
    ):
        assert field in blocked

    validation = validate_response_payload(normalized, source_prompt=request.prompt)
    assert validation.ok, validation.errors
