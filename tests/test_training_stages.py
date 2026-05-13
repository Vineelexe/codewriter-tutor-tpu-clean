from __future__ import annotations

import pytest

from cw360.train.stages import (
    ensure_trainable_stage,
    get_training_stage_spec,
    validate_training_stage,
)


def test_training_stage_specs_cover_phase9_stages() -> None:
    assert get_training_stage_spec("base_pretrain").trainable
    assert get_training_stage_spec("fim_train").requires_fim_format
    assert get_training_stage_spec("instruction_tune").instruction_weighted
    assert not get_training_stage_spec("eval_only").trainable


def test_training_stage_validation_rejects_unknown_stage() -> None:
    with pytest.raises(ValueError, match="training_stage"):
        validate_training_stage("general_chat")


def test_eval_only_is_not_trainable() -> None:
    with pytest.raises(ValueError, match="eval_only"):
        ensure_trainable_stage("eval_only")
