from __future__ import annotations

import json

from cw360.teacher.response_validation import validate_json_response, validate_response_payload
from tests.test_teacher_writer import _record


def test_response_validation_accepts_valid_record() -> None:
    result = validate_response_payload(_record().to_dict(), source_prompt="Fix Python code")

    assert result.ok, result.errors
    assert result.record is not None


def test_response_validation_rejects_invalid_json() -> None:
    result = validate_json_response("{not-json")

    assert not result.ok
    assert "invalid JSON" in result.errors[0]


def test_response_validation_rejects_fluff_and_missing_skills() -> None:
    payload = _record().to_dict()
    payload["skills"] = []
    payload["assistant_response"] = "As an AI language model, I hope this helps."

    result = validate_response_payload(payload, source_prompt="Python task")

    assert not result.ok
    assert "missing skills" in result.errors
    assert "generic fluff" in result.errors


def test_response_validation_rejects_provider_error_text() -> None:
    payload = _record().to_dict()
    payload["assistant_response"] = (
        "API error: unauthorized invalid api key for this Python request."
    )

    result = validate_json_response(json.dumps(payload), source_prompt="Python task")

    assert not result.ok
    assert "provider/API error text" in result.errors
