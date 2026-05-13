from __future__ import annotations

from itertools import count, islice

import pytest

from cw360.data.mixture import MixtureSampler, SourceExhaustedError
from cw360.data.schemas import TrainingExample


def _examples(source: str, n: int):
    for index in range(n):
        yield TrainingExample(
            text=f"def f_{source}_{index}():\n    return {index}\n",
            source=source,
            example_type="python_code",
            quality_tier="A",
        )


def _infinite(source: str):
    for index in count():
        yield TrainingExample(
            text=f"def f_{index}():\n    return {index}\n",
            source=source,
            example_type="python_code",
            quality_tier="A",
        )


def test_mixture_sampler_is_deterministic_and_tracks_counts() -> None:
    first = MixtureSampler(
        {"a": _examples("a", 10), "b": _examples("b", 10)},
        {"a": 0.75, "b": 0.25},
        seed=7,
    )
    second = MixtureSampler(
        {"a": _examples("a", 10), "b": _examples("b", 10)},
        {"a": 0.75, "b": 0.25},
        seed=7,
    )

    first_sources = [example.source for example in islice(first, 8)]
    second_sources = [example.source for example in islice(second, 8)]

    assert first_sources == second_sources
    assert first.stats().emitted_counts == {"a": 6, "b": 2}
    assert first.stats().total_emitted == 8


def test_mixture_sampler_handles_exhausted_finite_source_gracefully() -> None:
    sampler = MixtureSampler(
        {"finite": _examples("finite", 1), "infinite": _infinite("infinite")},
        {"finite": 0.9, "infinite": 0.1},
        seed=1,
        allow_exhausted=True,
    )

    emitted = list(islice(sampler, 5))

    assert [example.source for example in emitted].count("finite") == 1
    assert sampler.stats().exhausted_sources == ["finite"]
    assert sampler.stats().emitted_counts["infinite"] == 4


def test_mixture_sampler_can_fail_on_unexpected_exhaustion() -> None:
    sampler = MixtureSampler(
        {"finite": _examples("finite", 1), "other": _examples("other", 3)},
        {"finite": 0.9, "other": 0.1},
        seed=1,
        allow_exhausted=False,
    )

    with pytest.raises(SourceExhaustedError):
        list(sampler)


def test_mixture_sampler_downsamples_quality_tiers_and_scales_loss_weight() -> None:
    low = [
        TrainingExample(
            text="def low():\n    return 1\n",
            source="low",
            example_type="python_code",
            quality_tier="B",
            loss_weight=0.5,
        )
        for _ in range(4)
    ]
    high = [
        TrainingExample(
            text="def high():\n    return 1\n",
            source="high",
            example_type="python_code",
            quality_tier="A",
            loss_weight=1.0,
        )
    ]
    sampler = MixtureSampler(
        {"low": low, "high": high},
        {"low": 0.8, "high": 0.2},
        seed=3,
        tier_sampling_rates={"B": 0.0},
        tier_loss_multipliers={"A": 2.0},
    )

    emitted = list(sampler)

    assert [example.source for example in emitted] == ["high"]
    assert emitted[0].loss_weight == 2.0
    assert sampler.stats().skipped_by_tier["B"] == 4
