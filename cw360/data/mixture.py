from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TypeVar

from cw360.data.schemas import TrainingExample

T = TypeVar("T")


class SourceExhaustedError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class MixtureStats:
    target_weights: dict[str, float]
    target_ratios: dict[str, float]
    emitted_counts: dict[str, int]
    skipped_by_tier: dict[str, int]
    exhausted_sources: list[str]
    total_emitted: int

    @property
    def observed_ratios(self) -> dict[str, float]:
        if self.total_emitted == 0:
            return {name: 0.0 for name in self.target_weights}
        return {
            name: self.emitted_counts.get(name, 0) / self.total_emitted
            for name in self.target_weights
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "target_weights": dict(self.target_weights),
            "target_ratios": dict(self.target_ratios),
            "emitted_counts": dict(self.emitted_counts),
            "observed_ratios": self.observed_ratios,
            "skipped_by_tier": dict(self.skipped_by_tier),
            "exhausted_sources": list(self.exhausted_sources),
            "total_emitted": self.total_emitted,
        }


@dataclass(slots=True)
class MixtureSampler(Iterable[T]):
    sources: Mapping[str, Iterable[T]]
    weights: Mapping[str, float]
    seed: int = 0
    allow_exhausted: bool = True
    tier_sampling_rates: Mapping[str, float] | None = None
    tier_loss_multipliers: Mapping[str, float] | None = None
    _rng: random.Random = field(init=False, repr=False)
    _emitted_counts: Counter[str] = field(default_factory=Counter, init=False, repr=False)
    _skipped_by_tier: Counter[str] = field(default_factory=Counter, init=False, repr=False)
    _exhausted_sources: list[str] = field(default_factory=list, init=False, repr=False)
    _total_emitted: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError("at least one source iterator is required")
        missing = [name for name in self.weights if name not in self.sources]
        if missing:
            raise ValueError(f"weights include sources with no iterator: {missing}")
        positive_weights = {
            name: float(weight) for name, weight in self.weights.items() if weight > 0
        }
        if not positive_weights:
            raise ValueError("at least one target weight must be positive")
        object.__setattr__(self, "weights", positive_weights)
        object.__setattr__(self, "_rng", random.Random(self.seed))
        for tier, rate in (self.tier_sampling_rates or {}).items():
            if not 0.0 <= float(rate) <= 1.0:
                raise ValueError(f"tier sampling rate for {tier!r} must be between 0 and 1")

    @property
    def target_ratios(self) -> dict[str, float]:
        total = sum(self.weights.values())
        return {name: weight / total for name, weight in self.weights.items()}

    def __iter__(self) -> Iterator[T]:
        iterators: dict[str, Iterator[T]] = {
            name: iter(source)
            for name, source in self.sources.items()
            if self.weights.get(name, 0.0) > 0
        }
        active = set(iterators)

        while active:
            source_name = self._select_source(active)
            try:
                item = next(iterators[source_name])
            except StopIteration:
                self._mark_exhausted(source_name)
                active.remove(source_name)
                continue

            if not self._accept_quality_tier(item):
                continue

            self._apply_tier_multiplier(item)
            self._emitted_counts[source_name] += 1
            self._total_emitted += 1
            yield item

    def _select_source(self, active: set[str]) -> str:
        ratios = self.target_ratios
        next_total = self._total_emitted + 1
        deficits = {
            name: max(0.0, ratios[name] * next_total - self._emitted_counts.get(name, 0))
            for name in active
        }
        positive = {name: deficit for name, deficit in deficits.items() if deficit > 1.0e-12}
        weighted = positive or {name: ratios[name] for name in active}
        return self._weighted_choice(weighted)

    def _weighted_choice(self, weights: Mapping[str, float]) -> str:
        total = sum(weights.values())
        if total <= 0:
            return sorted(weights)[0]
        threshold = self._rng.random() * total
        cumulative = 0.0
        for name in sorted(weights):
            cumulative += weights[name]
            if threshold <= cumulative:
                return name
        return sorted(weights)[-1]

    def _mark_exhausted(self, source_name: str) -> None:
        if source_name not in self._exhausted_sources:
            self._exhausted_sources.append(source_name)
        if not self.allow_exhausted:
            raise SourceExhaustedError(f"source {source_name!r} exhausted")

    def _accept_quality_tier(self, item: T) -> bool:
        tier = _quality_tier(item)
        if tier is None:
            return True
        rate = float((self.tier_sampling_rates or {}).get(tier, 1.0))
        if rate >= 1.0:
            return True
        if self._rng.random() < rate:
            return True
        self._skipped_by_tier[tier] += 1
        return False

    def _apply_tier_multiplier(self, item: T) -> None:
        tier = _quality_tier(item)
        if tier is None or not isinstance(item, TrainingExample):
            return
        multipliers = self.tier_loss_multipliers or {}
        if tier not in multipliers:
            return
        base = item.loss_weight if item.loss_weight is not None else 1.0
        item.loss_weight = base * float(multipliers[tier])

    def stats(self) -> MixtureStats:
        return MixtureStats(
            target_weights=dict(self.weights),
            target_ratios=self.target_ratios,
            emitted_counts={name: self._emitted_counts.get(name, 0) for name in self.weights},
            skipped_by_tier=dict(self._skipped_by_tier),
            exhausted_sources=list(self._exhausted_sources),
            total_emitted=self._total_emitted,
        )


def _quality_tier(item: object) -> str | None:
    tier = getattr(item, "quality_tier", None)
    if isinstance(tier, str) and tier:
        return tier
    return None
