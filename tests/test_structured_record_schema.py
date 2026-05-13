from __future__ import annotations

import pytest

from cw360.teacher.structured_record import StructuredTrainingRecord
from tests.test_teacher_writer import _record


def test_structured_record_round_trips_required_schema() -> None:
    record = _record()
    payload = record.to_dict()

    reloaded = StructuredTrainingRecord.from_dict(payload)

    assert reloaded.to_dict() == payload
    assert "User:" in reloaded.to_training_text()
    assert "Assistant:" in reloaded.to_training_text()


def test_structured_record_rejects_invalid_stage() -> None:
    payload = _record().to_dict()
    payload["train_stage"] = "chatbot"

    with pytest.raises(ValueError, match="invalid train_stage"):
        StructuredTrainingRecord.from_dict(payload)
