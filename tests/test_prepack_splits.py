from __future__ import annotations

import pytest

from cw360.prepack.shard_index import assign_split_for_key, stable_bucket


def test_stable_split_assignment_is_repeatable() -> None:
    first = assign_split_for_key(
        "candidate:abc",
        train_fraction=0.8,
        val_fraction=0.1,
        test_fraction=0.1,
    )
    second = assign_split_for_key(
        "candidate:abc",
        train_fraction=0.8,
        val_fraction=0.1,
        test_fraction=0.1,
    )

    assert first == second
    assert 0.0 <= stable_bucket("candidate:abc") <= 1.0


def test_split_assignment_rejects_bad_fractions() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        assign_split_for_key(
            "candidate:abc",
            train_fraction=0.8,
            val_fraction=0.2,
            test_fraction=0.2,
        )
