from __future__ import annotations

from cw360.teacher.prompts import build_teacher_prompt
from cw360.teacher.providers.mock import MockTeacherClient
from cw360.teacher.response_validation import validate_json_response
from tests.test_teacher_prompt_builders import _candidate


def test_mock_provider_returns_valid_structured_record() -> None:
    client = MockTeacherClient()
    bundle = build_teacher_prompt(_candidate("tests"))

    response = client.generate(bundle.prompt, bundle.system_prompt, 0.2, 1200)
    validation = validate_json_response(response.text, source_prompt=bundle.prompt)

    assert client.calls == 1
    assert validation.ok, validation.errors
    assert validation.record is not None
    assert validation.record.teacher_provider == "mock"
    assert validation.record.task_type == "tests"
