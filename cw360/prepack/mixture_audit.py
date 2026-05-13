from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from cw360.prepack.manifest import normalize_counts


class MixtureAuditError(AssertionError):
    pass


@dataclass(frozen=True, slots=True)
class MixtureAuditResult:
    target: dict[str, float]
    actual: dict[str, float]
    tolerance_abs: float
    max_abs_error: float


def normalized_target(weights: Mapping[str, float]) -> dict[str, float]:
    positive = {str(name): float(weight) for name, weight in weights.items() if weight > 0}
    return normalize_counts(positive)


def audit_mixture(
    *,
    actual_counts: Mapping[str, int | float],
    target_weights: Mapping[str, float],
    tolerance_abs: float,
    label: str,
) -> MixtureAuditResult:
    target = normalized_target(target_weights)
    actual = normalize_counts(actual_counts)
    missing = set(target).difference(actual)
    for name in missing:
        actual[name] = 0.0
    max_error = 0.0
    for name, expected in target.items():
        max_error = max(max_error, abs(actual.get(name, 0.0) - expected))
    if max_error > tolerance_abs:
        raise MixtureAuditError(
            f"{label} mixture differs from target by {max_error:.6f}, "
            f"tolerance is {tolerance_abs:.6f}"
        )
    return MixtureAuditResult(
        target=target,
        actual=actual,
        tolerance_abs=tolerance_abs,
        max_abs_error=max_error,
    )
