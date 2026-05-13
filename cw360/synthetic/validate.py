from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cw360.synthetic.quality import quality_rejection_reasons
from cw360.synthetic.schemas import (
    StructuredSyntheticRecord,
    SyntheticSchemaError,
    validate_enum_fields,
)


@dataclass(frozen=True, slots=True)
class ValidationResult:
    record: StructuredSyntheticRecord | None
    valid: bool
    reasons: list[str]
    line_number: int | None = None
    raw: dict[str, Any] | None = None


def validate_record_payload(
    payload: Mapping[str, Any],
    *,
    line_number: int | None = None,
) -> ValidationResult:
    raw = dict(payload)
    try:
        record = StructuredSyntheticRecord.from_dict(raw)
    except SyntheticSchemaError as exc:
        return ValidationResult(
            record=None,
            valid=False,
            reasons=[str(exc)],
            line_number=line_number,
            raw=raw,
        )

    reasons = _required_semantic_reasons(record)
    reasons.extend(validate_enum_fields(record))
    reasons.extend(quality_rejection_reasons(record))
    return ValidationResult(
        record=record if not reasons else None,
        valid=not reasons,
        reasons=reasons,
        line_number=line_number,
        raw=raw,
    )


def iter_validation_results(path: str | Path) -> Iterator[ValidationResult]:
    jsonl_path = Path(path)
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                yield ValidationResult(
                    record=None,
                    valid=False,
                    reasons=[f"malformed_json:{exc.msg}"],
                    line_number=line_number,
                    raw=None,
                )
                continue
            if not isinstance(payload, dict):
                yield ValidationResult(
                    record=None,
                    valid=False,
                    reasons=["malformed_json:expected_object"],
                    line_number=line_number,
                    raw=None,
                )
                continue
            yield validate_record_payload(payload, line_number=line_number)


def valid_records_from_results(
    results: Iterable[ValidationResult],
) -> list[StructuredSyntheticRecord]:
    return [result.record for result in results if result.valid and result.record is not None]


def _required_semantic_reasons(record: StructuredSyntheticRecord) -> list[str]:
    reasons: list[str] = []
    if not (record.source_candidate_id or record.candidate_id):
        reasons.append("missing_candidate_id")
    if not record.skills:
        reasons.append("missing_skills")
    if not 0.0 < float(record.loss_weight) <= 3.0:
        reasons.append("invalid_loss_weight")
    return reasons
