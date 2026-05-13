from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QualityTier(str, Enum):
    TIER_A = "A"
    TIER_B = "B"
    TIER_C = "C"
    RESCUE = "RESCUE"


@dataclass(slots=True)
class TrainingExample:
    text: str
    source: str
    example_type: str
    metadata: dict[str, Any] = field(default_factory=dict)
    quality_tier: str | None = None
    loss_weight: float | None = None
    training_stage: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("TrainingExample.text must be a string")
        if not self.source:
            raise ValueError("TrainingExample.source must be non-empty")
        if not self.example_type:
            raise ValueError("TrainingExample.example_type must be non-empty")
        if self.loss_weight is not None and self.loss_weight < 0:
            raise ValueError("TrainingExample.loss_weight must be non-negative")


@dataclass(frozen=True, slots=True)
class FilterDecision:
    accepted: bool
    quality_tier: QualityTier
    reason: str
    loss_weight: float
    rescue: bool = False

    @property
    def status(self) -> str:
        if self.rescue:
            return "rescued"
        return "accepted" if self.accepted else "rejected"


@dataclass(frozen=True, slots=True)
class ExtractedRecord:
    example: TrainingExample | None
    error: str | None = None
    line_number: int | None = None

    @property
    def ok(self) -> bool:
        return self.example is not None and self.error is None
