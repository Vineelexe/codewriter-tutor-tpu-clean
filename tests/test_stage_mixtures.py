from __future__ import annotations

import pytest

from cw360.data.modes import REAL_TPU_TRAINING_INPUT, mode_is_tpu_training_path, parse_pipeline_mode
from cw360.data.stage_mixtures import BASE_TARGET_WEIGHTS, get_stage_mixture


def test_base_pretrain_mixture_matches_phase8_target_ratios() -> None:
    mixture = get_stage_mixture("base_pretrain")

    assert dict(mixture.weights) == BASE_TARGET_WEIGHTS
    assert mixture.cheap_base_ratio == pytest.approx(0.65)
    assert mixture.debug_instruction_ratio == pytest.approx(0.35)


def test_all_stage_mixtures_are_normalized_and_stage_specific() -> None:
    assert get_stage_mixture("fim_train").normalized_weights["stack_v2_python"] > 0.4
    assert get_stage_mixture("instruction_tune").normalized_weights["synthetic"] == 0.45
    assert get_stage_mixture("eval_only").normalized_weights["debugbench"] == 0.40


def test_modes_do_not_define_live_streaming_as_tpu_training_path() -> None:
    assert parse_pipeline_mode("prepack_stream").value == "prepack_stream"
    assert not mode_is_tpu_training_path("prepack_stream")
    assert REAL_TPU_TRAINING_INPUT == "prepacked_token_shards"


def test_unknown_stage_is_rejected() -> None:
    with pytest.raises(ValueError, match="training_stage"):
        get_stage_mixture("train_streaming")
