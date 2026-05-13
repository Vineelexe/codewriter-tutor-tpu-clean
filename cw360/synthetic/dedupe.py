from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from cw360.synthetic.schemas import StructuredSyntheticRecord


@dataclass(frozen=True, slots=True)
class DuplicateRecord:
    synthetic_id: str
    duplicate_of: str
    reason: str


@dataclass(slots=True)
class SyntheticDeduper:
    allow_duplicates: bool = False
    _seen: dict[tuple[str, str, str, str], StructuredSyntheticRecord] = field(
        default_factory=dict,
        init=False,
    )

    def check(self, record: StructuredSyntheticRecord) -> DuplicateRecord | None:
        key = duplicate_key(record)
        previous = self._seen.get(key)
        if previous is None:
            self._seen[key] = record
            return None
        if self.allow_duplicates:
            return None
        return DuplicateRecord(
            synthetic_id=record.synthetic_id,
            duplicate_of=previous.synthetic_id,
            reason="duplicate_candidate_task_provider_output",
        )


@dataclass(frozen=True, slots=True)
class DedupeResult:
    records: list[StructuredSyntheticRecord]
    duplicates: list[DuplicateRecord]


def dedupe_records(
    records: Iterable[StructuredSyntheticRecord],
    *,
    allow_duplicates: bool = False,
) -> DedupeResult:
    deduper = SyntheticDeduper(allow_duplicates=allow_duplicates)
    accepted: list[StructuredSyntheticRecord] = []
    duplicates: list[DuplicateRecord] = []
    for record in records:
        duplicate = deduper.check(record)
        if duplicate is None:
            accepted.append(record)
        else:
            duplicates.append(duplicate)
    return DedupeResult(records=accepted, duplicates=duplicates)


def duplicate_key(record: StructuredSyntheticRecord) -> tuple[str, str, str, str]:
    candidate_key = record.source_candidate_id or record.candidate_id or record.synthetic_id
    output_key = record.output_hash or _normalized(record.assistant_response)
    return (
        candidate_key,
        record.task_type,
        record.teacher_provider,
        output_key,
    )


def _normalized(value: str) -> str:
    return " ".join(value.lower().split())
