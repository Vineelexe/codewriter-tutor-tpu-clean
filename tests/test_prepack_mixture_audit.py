from __future__ import annotations

import pytest

from cw360.prepack.mixture_audit import MixtureAuditError, audit_mixture


def test_mixture_audit_accepts_counts_within_tolerance() -> None:
    result = audit_mixture(
        actual_counts={"a": 51, "b": 49},
        target_weights={"a": 0.5, "b": 0.5},
        tolerance_abs=0.02,
        label="unit",
    )

    assert result.max_abs_error == pytest.approx(0.01)


def test_mixture_audit_rejects_counts_outside_tolerance() -> None:
    with pytest.raises(MixtureAuditError):
        audit_mixture(
            actual_counts={"a": 80, "b": 20},
            target_weights={"a": 0.5, "b": 0.5},
            tolerance_abs=0.05,
            label="unit",
        )
