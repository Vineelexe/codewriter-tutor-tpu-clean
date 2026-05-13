from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from cw360.synthetic.dedupe import DuplicateRecord
from cw360.synthetic.schemas import StructuredSyntheticRecord
from cw360.synthetic.validate import ValidationResult


@dataclass(frozen=True, slots=True)
class SyntheticDataReport:
    total_records: int
    valid_records: int
    rejected_records: int
    duplicate_records: int
    count_by_task_type: dict[str, int]
    count_by_source: dict[str, int]
    count_by_provider_model: dict[str, int]
    count_by_difficulty: dict[str, int]
    count_by_train_stage: dict[str, int]
    average_loss_weight: float
    rejection_reasons: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "rejected_records": self.rejected_records,
            "duplicate_records": self.duplicate_records,
            "count_by_task_type": self.count_by_task_type,
            "count_by_source": self.count_by_source,
            "count_by_provider_model": self.count_by_provider_model,
            "count_by_difficulty": self.count_by_difficulty,
            "count_by_train_stage": self.count_by_train_stage,
            "average_loss_weight": self.average_loss_weight,
            "rejection_reasons": self.rejection_reasons,
        }


@dataclass(slots=True)
class ReportBuilder:
    validation_results: Iterable[ValidationResult]
    duplicates: Iterable[DuplicateRecord] = field(default_factory=list)

    def build(self) -> SyntheticDataReport:
        results = list(self.validation_results)
        duplicate_rows = list(self.duplicates)
        valid = [result.record for result in results if result.valid and result.record is not None]
        rejection_reasons: Counter[str] = Counter()
        for result in results:
            if result.valid:
                continue
            rejection_reasons.update(result.reasons)
        rejection_reasons.update(duplicate.reason for duplicate in duplicate_rows)
        return SyntheticDataReport(
            total_records=len(results),
            valid_records=max(0, len(valid) - len(duplicate_rows)),
            rejected_records=sum(1 for result in results if not result.valid),
            duplicate_records=len(duplicate_rows),
            count_by_task_type=_count(valid, "task_type"),
            count_by_source=_count(valid, "source"),
            count_by_provider_model=_count_provider_model(valid),
            count_by_difficulty=_count(valid, "difficulty"),
            count_by_train_stage=_count(valid, "train_stage"),
            average_loss_weight=_average_loss_weight(valid),
            rejection_reasons=dict(rejection_reasons),
        )


def build_report(
    validation_results: Iterable[ValidationResult],
    duplicates: Iterable[DuplicateRecord] = (),
) -> SyntheticDataReport:
    return ReportBuilder(validation_results, duplicates).build()


def _count(records: list[StructuredSyntheticRecord], attribute: str) -> dict[str, int]:
    return dict(Counter(str(getattr(record, attribute)) for record in records))


def _count_provider_model(records: list[StructuredSyntheticRecord]) -> dict[str, int]:
    return dict(
        Counter(f"{record.teacher_provider}/{record.teacher_model}" for record in records)
    )


def _average_loss_weight(records: list[StructuredSyntheticRecord]) -> float:
    if not records:
        return 0.0
    return sum(float(record.loss_weight) for record in records) / len(records)
