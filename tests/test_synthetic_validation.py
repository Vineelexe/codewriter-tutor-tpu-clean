from __future__ import annotations

from pathlib import Path

from cw360.synthetic.validate import iter_validation_results, validate_record_payload
from tests.test_synthetic_schema import sample_payload


def test_validation_accepts_python_structured_record() -> None:
    result = validate_record_payload(sample_payload())

    assert result.valid
    assert result.record is not None
    assert result.reasons == []


def test_validation_rejects_unknown_task_and_missing_candidate() -> None:
    payload = sample_payload(task_type="general_chat", source_candidate_id="", candidate_id=None)

    result = validate_record_payload(payload)

    assert not result.valid
    assert "invalid_task_type" in result.reasons
    assert "missing_candidate_id" in result.reasons


def test_validation_rejects_provider_error_and_near_copy() -> None:
    prompt = "Debug this Python function that raises ValueError in parse_int."
    payload = sample_payload(
        user_prompt=prompt,
        assistant_response=prompt,
        output_hash="copy-output",
    )

    result = validate_record_payload(payload)

    assert not result.valid
    assert "response_nearly_identical_to_prompt" in result.reasons


def test_validation_reports_malformed_json_line(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.jsonl"
    path.write_text("not-json\n", encoding="utf-8")

    result = list(iter_validation_results(path))[0]

    assert not result.valid
    assert result.reasons[0].startswith("malformed_json")
