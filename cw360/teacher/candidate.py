from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TASK_TYPES = frozenset(
    {"code_writing", "debugging", "explanation", "comments", "refactor", "tests"}
)
QUALITY_TIERS = frozenset({"A", "B", "C", "RESCUE"})


@dataclass(frozen=True, slots=True)
class TeacherCandidate:
    candidate_id: str
    source: str
    task_type: str
    score: float
    input_hash: str
    estimated_tokens: int
    reason_selected: str
    quality_tier: str
    proposed_train_stage: str
    proposed_loss_weight: float
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_text: str | None = None
    selected_text: str | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must be non-empty")
        if not self.source:
            raise ValueError("source must be non-empty")
        if self.task_type not in TASK_TYPES:
            raise ValueError(f"unsupported task_type: {self.task_type}")
        if self.quality_tier not in QUALITY_TIERS:
            raise ValueError(f"unsupported quality_tier: {self.quality_tier}")
        if self.raw_text is None and self.selected_text is None:
            raise ValueError("candidate must include raw_text or selected_text")
        if self.estimated_tokens < 0:
            raise ValueError("estimated_tokens must be non-negative")
        if self.proposed_loss_weight < 0:
            raise ValueError("proposed_loss_weight must be non-negative")

    @property
    def text(self) -> str:
        return self.selected_text if self.selected_text is not None else str(self.raw_text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source": self.source,
            "task_type": self.task_type,
            "raw_text": self.raw_text,
            "selected_text": self.selected_text,
            "score": self.score,
            "input_hash": self.input_hash,
            "estimated_tokens": self.estimated_tokens,
            "metadata": dict(self.metadata),
            "reason_selected": self.reason_selected,
            "quality_tier": self.quality_tier,
            "proposed_train_stage": self.proposed_train_stage,
            "proposed_loss_weight": self.proposed_loss_weight,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TeacherCandidate:
        return cls(
            candidate_id=str(payload["candidate_id"]),
            source=str(payload["source"]),
            task_type=str(payload["task_type"]),
            raw_text=payload.get("raw_text") if isinstance(payload.get("raw_text"), str) else None,
            selected_text=payload.get("selected_text")
            if isinstance(payload.get("selected_text"), str)
            else None,
            score=float(payload["score"]),
            input_hash=str(payload["input_hash"]),
            estimated_tokens=int(payload["estimated_tokens"]),
            metadata=dict(payload.get("metadata") or {}),
            reason_selected=str(payload["reason_selected"]),
            quality_tier=str(payload["quality_tier"]),
            proposed_train_stage=str(payload["proposed_train_stage"]),
            proposed_loss_weight=float(payload["proposed_loss_weight"]),
        )
