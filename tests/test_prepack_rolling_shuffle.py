from __future__ import annotations

from cw360.prepack.rolling_shuffle import rolling_shuffle


def test_rolling_shuffle_is_deterministic_and_breaks_source_runs() -> None:
    items = ["stack"] * 10 + ["synthetic"] * 10

    first = list(rolling_shuffle(items, buffer_size=5, seed=13))
    second = list(rolling_shuffle(items, buffer_size=5, seed=13))

    assert first == second
    assert first != items
    assert set(first) == {"stack", "synthetic"}
    assert first[:10] != ["stack"] * 10
