from __future__ import annotations

from cw360.data.quality_tiers import apply_decision, tier_from_signal
from cw360.data.schemas import QualityTier, TrainingExample


def test_tier_a_is_high_weight_and_accepted() -> None:
    decision = tier_from_signal(high_signal=True, reason="dense_pair")
    assert decision.accepted
    assert decision.quality_tier is QualityTier.TIER_A
    assert decision.loss_weight == 1.0


def test_tier_b_is_light_weight_and_accepted() -> None:
    decision = tier_from_signal(useful=True, reason="imperfect")
    assert decision.accepted
    assert decision.quality_tier is QualityTier.TIER_B
    assert 0 < decision.loss_weight < 1


def test_rescue_is_not_accepted_for_training_but_not_trash() -> None:
    decision = tier_from_signal(rescue=True, reason="long_advanced")
    assert not decision.accepted
    assert decision.rescue
    assert decision.status == "rescued"


def test_apply_decision_updates_training_example_metadata() -> None:
    example = TrainingExample(text="def f():\n    return 1\n", source="synthetic", example_type="x")
    apply_decision(example, tier_from_signal(high_signal=True, reason="ok"))
    assert example.quality_tier == "A"
    assert example.metadata["filter_status"] == "accepted"
