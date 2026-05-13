from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from cw360.constants import VALID_TRAINING_STAGES

TASK_TYPES = frozenset(
    {"code_writing", "debugging", "explanation", "comments", "refactor", "tests"}
)
DIFFICULTIES = frozenset({"beginner", "intermediate", "advanced"})
RECORD_TYPES = frozenset({"structured", "synthetic"})


class SyntheticSchemaError(ValueError):
    pass


@dataclass(slots=True)
class StructuredSyntheticRecord:
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
    candidate_id: str | None = None
    code_before: str | None = None
    code_after: str | None = None
    explanation: str | None = None
    tests: str | None = None
    quality_notes: str | None = None

    @property
    def stable_candidate_key(self) -> str:
        return self.source_candidate_id or self.candidate_id or self.synthetic_id

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: value for key, value in payload.items() if value is not None}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StructuredSyntheticRecord:
        try:
            skills = payload["skills"]
            if not isinstance(skills, list):
                raise SyntheticSchemaError("skills must be a list")
            return cls(
                synthetic_id=_required_str(payload, "synthetic_id"),
                source_candidate_id=str(payload.get("source_candidate_id") or ""),
                candidate_id=_optional_str(payload, "candidate_id"),
                source=_required_str(payload, "source"),
                record_type=_required_str(payload, "record_type"),
                task_type=_required_str(payload, "task_type"),
                difficulty=_required_str(payload, "difficulty"),
                skills=[str(skill) for skill in skills if str(skill).strip()],
                user_prompt=_required_str(payload, "user_prompt"),
                assistant_response=_required_str(payload, "assistant_response"),
                train_stage=_required_str(payload, "train_stage"),
                loss_weight=float(payload["loss_weight"]),
                teacher_provider=_required_str(payload, "teacher_provider"),
                teacher_model=_required_str(payload, "teacher_model"),
                generation_time=_required_str(payload, "generation_time"),
                input_hash=_required_str(payload, "input_hash"),
                output_hash=_required_str(payload, "output_hash"),
                metadata=dict(payload.get("metadata") or {}),
                code_before=_optional_str(payload, "code_before"),
                code_after=_optional_str(payload, "code_after"),
                explanation=_optional_str(payload, "explanation"),
                tests=_optional_str(payload, "tests"),
                quality_notes=_optional_str(payload, "quality_notes"),
            )
        except KeyError as exc:
            raise SyntheticSchemaError(f"missing required field: {exc.args[0]}") from exc
        except (TypeError, ValueError) as exc:
            if isinstance(exc, SyntheticSchemaError):
                raise
            raise SyntheticSchemaError(str(exc)) from exc


def stable_hash(value: str | bytes) -> str:
    data = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def stable_json_hash(payload: dict[str, Any]) -> str:
    return stable_hash(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value.strip():
        raise SyntheticSchemaError(f"{key} must be a non-empty string")
    return value


def _optional_str(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise SyntheticSchemaError(f"{key} must be a string when present")
    return value


def validate_enum_fields(record: StructuredSyntheticRecord) -> list[str]:
    reasons: list[str] = []
    if record.record_type not in RECORD_TYPES:
        reasons.append("invalid_record_type")
    if record.task_type not in TASK_TYPES:
        reasons.append("invalid_task_type")
    if record.difficulty not in DIFFICULTIES:
        reasons.append("invalid_difficulty")
    if record.train_stage not in VALID_TRAINING_STAGES:
        reasons.append("invalid_train_stage")
    return reasons
