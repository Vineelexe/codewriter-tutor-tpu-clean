from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from cw360.data.extractors import extract_training_example
from cw360.data.filters import estimate_token_count
from cw360.data.schemas import TrainingExample
from cw360.teacher.candidate import TASK_TYPES, TeacherCandidate
from cw360.teacher.candidate_score import score_teacher_candidate
from cw360.utils.hashing import sha256_bytes


@dataclass(frozen=True, slots=True)
class TeacherCandidateConfig:
    min_score: float = 0.70
    max_input_tokens: int = 1600
    min_input_tokens: int = 80
    max_candidates_per_source: dict[str, int] = field(default_factory=dict)
    max_candidates_per_task_type: dict[str, int] = field(default_factory=dict)
    task_mix: dict[str, float] = field(default_factory=dict)
    output_format: str = "jsonl"
    dataset_config: str = "configs/datasets.yaml"

    @classmethod
    def from_yaml(cls, path: str | Path) -> TeacherCandidateConfig:
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"teacher candidate config {config_path} must be a mapping")
        return cls(
            min_score=float(raw.get("min_score", 0.70)),
            max_input_tokens=int(raw.get("max_input_tokens", 1600)),
            min_input_tokens=int(raw.get("min_input_tokens", 80)),
            max_candidates_per_source={
                str(key): int(value)
                for key, value in dict(raw.get("max_candidates_per_source") or {}).items()
            },
            max_candidates_per_task_type={
                str(key): int(value)
                for key, value in dict(raw.get("max_candidates_per_task_type") or {}).items()
            },
            task_mix={
                str(key): float(value)
                for key, value in dict(raw.get("task_mix") or {}).items()
            },
            output_format=str(raw.get("output_format", "jsonl")),
            dataset_config=str(raw.get("dataset_config", "configs/datasets.yaml")),
        )


@dataclass(slots=True)
class CandidateRouter:
    config: TeacherCandidateConfig
    max_candidates: int | None = None
    min_score_override: float | None = None
    source_counts: Counter[str] = field(default_factory=Counter)
    task_counts: Counter[str] = field(default_factory=Counter)
    total_selected: int = 0
    rejected: int = 0

    def route_example(
        self,
        item: TrainingExample | Mapping[str, Any],
        *,
        source: str,
    ) -> TeacherCandidate | None:
        if self.is_complete:
            return None
        if self._source_cap_reached(source):
            self.rejected += 1
            return None

        example = _coerce_example(item, source=source)
        text = example.text
        score = score_teacher_candidate(
            source=example.source,
            text=text,
            metadata=example.metadata,
            min_score=self.min_score_override
            if self.min_score_override is not None
            else self.config.min_score,
            min_input_tokens=self.config.min_input_tokens,
            max_input_tokens=self.config.max_input_tokens,
        )
        if not score.accepted:
            self.rejected += 1
            return None

        task_type = self._choose_task(score.task_preferences)
        if task_type is None:
            self.rejected += 1
            return None

        input_hash = sha256_bytes(
            "\0".join([example.source, task_type, score.selected_text]).encode("utf-8")
        )
        ordinal = self.total_selected + 1
        candidate = TeacherCandidate(
            candidate_id=f"{example.source}:{task_type}:{input_hash[:16]}:{ordinal:06d}",
            source=example.source,
            task_type=task_type,
            raw_text=None if score.selected_text != text else text,
            selected_text=score.selected_text if score.selected_text != text else None,
            score=score.score,
            input_hash=input_hash,
            estimated_tokens=score.estimated_tokens or estimate_token_count(score.selected_text),
            metadata={**example.metadata, **score.metadata},
            reason_selected=score.reason,
            quality_tier=score.quality_tier,
            proposed_train_stage=score.proposed_train_stage,
            proposed_loss_weight=score.proposed_loss_weight,
        )
        self.source_counts[candidate.source] += 1
        self.task_counts[candidate.task_type] += 1
        self.total_selected += 1
        return candidate

    def route_many(
        self,
        items: Iterable[TrainingExample | Mapping[str, Any]],
        *,
        source: str,
        limit: int | None = None,
    ) -> Iterator[TeacherCandidate]:
        seen = 0
        for item in items:
            if limit is not None and seen >= limit:
                return
            seen += 1
            if self.is_complete:
                return
            candidate = self.route_example(item, source=source)
            if candidate is not None:
                yield candidate

    @property
    def is_complete(self) -> bool:
        return self.max_candidates is not None and self.total_selected >= self.max_candidates

    def _choose_task(self, preferences: tuple[str, ...]) -> str | None:
        for task_type in preferences:
            if task_type not in TASK_TYPES:
                continue
            if not self._task_cap_reached(task_type):
                return task_type
        for task_type in sorted(TASK_TYPES):
            if not self._task_cap_reached(task_type):
                return task_type
        return None

    def _source_cap_reached(self, source: str) -> bool:
        cap = self.config.max_candidates_per_source.get(source)
        return cap is not None and self.source_counts[source] >= cap

    def _task_cap_reached(self, task_type: str) -> bool:
        cap = self._task_cap(task_type)
        return cap is not None and self.task_counts[task_type] >= cap

    def _task_cap(self, task_type: str) -> int | None:
        explicit = self.config.max_candidates_per_task_type.get(task_type)
        if explicit is not None:
            return explicit
        if self.max_candidates is None:
            return None
        mix = self.config.task_mix.get(task_type)
        if mix is None:
            return None
        return max(1, round(self.max_candidates * mix))


def _coerce_example(item: TrainingExample | Mapping[str, Any], *, source: str) -> TrainingExample:
    if isinstance(item, TrainingExample):
        return item
    return extract_training_example(source, item)
