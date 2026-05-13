from __future__ import annotations

from cw360.data.schemas import FilterDecision, QualityTier, TrainingExample

DEFAULT_LOSS_WEIGHTS: dict[QualityTier, float] = {
    QualityTier.TIER_A: 1.0,
    QualityTier.TIER_B: 0.35,
    QualityTier.TIER_C: 0.0,
    QualityTier.RESCUE: 0.15,
}


def decision_for_tier(tier: QualityTier, reason: str) -> FilterDecision:
    return FilterDecision(
        accepted=tier in {QualityTier.TIER_A, QualityTier.TIER_B},
        rescue=tier is QualityTier.RESCUE,
        quality_tier=tier,
        reason=reason,
        loss_weight=DEFAULT_LOSS_WEIGHTS[tier],
    )


def apply_decision(example: TrainingExample, decision: FilterDecision) -> TrainingExample:
    example.quality_tier = decision.quality_tier.value
    example.loss_weight = decision.loss_weight
    example.metadata["filter_reason"] = decision.reason
    example.metadata["filter_status"] = decision.status
    return example


def tier_from_signal(
    *,
    high_signal: bool = False,
    useful: bool = True,
    rescue: bool = False,
    trash: bool = False,
    reason: str = "quality_heuristic",
) -> FilterDecision:
    if trash:
        return decision_for_tier(QualityTier.TIER_C, reason)
    if rescue:
        return decision_for_tier(QualityTier.RESCUE, reason)
    if high_signal:
        return decision_for_tier(QualityTier.TIER_A, reason)
    if useful:
        return decision_for_tier(QualityTier.TIER_B, reason)
    return decision_for_tier(QualityTier.TIER_C, reason)
