from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from cw360.constants import VALID_TRAINING_STAGES

DIFFICULTIES = frozenset({"beginner", "intermediate", "advanced"})
RECORD_TYPES = frozenset({"structured", "synthetic"})
TASK_TYPES = frozenset(
    {"code_writing", "debugging", "explanation", "comments", "refactor", "tests"}
)
VALID_LOSS_WEIGHTS = frozenset({0.5, 1.0, 1.5})


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_hash(value: str | bytes) -> str:
    data = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def stable_json_hash(payload: dict[str, Any]) -> str:
    return stable_hash(json.dumps(payload, sort_keys=True, separators=(",", ":")))


@dataclass(slots=True)
class StructuredTrainingRecord:
    synthetic_id: str
    source_candidate_id: str
    source: str
    record_type: str
    task_type: str
    difficulty: str
    skills: list[str]
    user_prompt: str
    assistant_response: str
    train_stage: str
    loss_weight: float
    teacher_provider: str
    teacher_model: str
    generation_time: str
    input_hash: str
    output_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    code_before: str | None = None
    code_after: str | None = None
    explanation: str | None = None
    tests: str | None = None
    quality_notes: str | None = None

    def __post_init__(self) -> None:
        if not self.synthetic_id:
            raise ValueError("synthetic_id must be non-empty")
        if not self.source_candidate_id:
            raise ValueError("source_candidate_id must be non-empty")
        if not self.source:
            raise ValueError("source must be non-empty")
        if self.record_type not in RECORD_TYPES:
            raise ValueError(f"invalid record_type: {self.record_type}")
        if self.task_type not in TASK_TYPES:
            raise ValueError(f"invalid task_type: {self.task_type}")
        if self.difficulty not in DIFFICULTIES:
            raise ValueError(f"invalid difficulty: {self.difficulty}")
        if not self.skills:
            raise ValueError("skills must be non-empty")
        if not self.user_prompt.strip():
            raise ValueError("user_prompt must be non-empty")
        if not self.assistant_response.strip():
            raise ValueError("assistant_response must be non-empty")
        if self.train_stage not in VALID_TRAINING_STAGES:
            raise ValueError(f"invalid train_stage: {self.train_stage}")
        if float(self.loss_weight) not in VALID_LOSS_WEIGHTS:
            raise ValueError("loss_weight must be one of 0.5, 1.0, or 1.5")
        if not self.teacher_provider:
            raise ValueError("teacher_provider must be non-empty")
        if not self.teacher_model:
            raise ValueError("teacher_model must be non-empty")
        if not self.input_hash:
            raise ValueError("input_hash must be non-empty")
        if not self.output_hash:
            raise ValueError("output_hash must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "synthetic_id": self.synthetic_id,
            "source_candidate_id": self.source_candidate_id,
            "source": self.source,
            "record_type": self.record_type,
            "task_type": self.task_type,
            "difficulty": self.difficulty,
            "skills": list(self.skills),
            "user_prompt": self.user_prompt,
            "assistant_response": self.assistant_response,
            "train_stage": self.train_stage,
            "loss_weight": float(self.loss_weight),
            "teacher_provider": self.teacher_provider,
            "teacher_model": self.teacher_model,
            "generation_time": self.generation_time,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "metadata": dict(self.metadata),
        }
        for key in ("code_before", "code_after", "explanation", "tests", "quality_notes"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StructuredTrainingRecord:
        return cls(
            synthetic_id=str(payload["synthetic_id"]),
            source_candidate_id=str(payload["source_candidate_id"]),
            source=str(payload["source"]),
            record_type=str(payload["record_type"]),
            task_type=str(payload["task_type"]),
            difficulty=str(payload["difficulty"]),
            skills=[str(skill) for skill in payload["skills"]],
            user_prompt=str(payload["user_prompt"]),
            assistant_response=str(payload["assistant_response"]),
            train_stage=str(payload["train_stage"]),
            loss_weight=float(payload["loss_weight"]),
            teacher_provider=str(payload["teacher_provider"]),
            teacher_model=str(payload["teacher_model"]),
            generation_time=str(payload["generation_time"]),
            input_hash=str(payload["input_hash"]),
            output_hash=str(payload["output_hash"]),
            metadata=dict(payload.get("metadata") or {}),
            code_before=payload.get("code_before"),
            code_after=payload.get("code_after"),
            explanation=payload.get("explanation"),
            tests=payload.get("tests"),
            quality_notes=payload.get("quality_notes"),
        )

    def to_training_text(self) -> str:
        user = self.user_prompt.strip()
        assistant = self.assistant_response.strip()
        return f"User:\n{user}\n\nAssistant:\n{assistant}\n"


def make_synthetic_id(source_candidate_id: str, request_hash: str, output_hash: str) -> str:
    digest = stable_hash(f"{source_candidate_id}:{request_hash}:{output_hash}")[:20]
    return f"cw360-synth-{digest}"
