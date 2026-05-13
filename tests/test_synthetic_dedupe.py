from __future__ import annotations

from cw360.synthetic.dedupe import dedupe_records
from cw360.synthetic.schemas import StructuredSyntheticRecord
from tests.test_synthetic_schema import sample_payload


def _record(identifier: str, **overrides) -> StructuredSyntheticRecord:
    return StructuredSyntheticRecord.from_dict(sample_payload(synthetic_id=identifier, **overrides))


def test_dedupe_rejects_duplicate_candidate_task_provider_output() -> None:
    records = [_record("synth-1"), _record("synth-2")]

    result = dedupe_records(records)

    assert [record.synthetic_id for record in result.records] == ["synth-1"]
    assert result.duplicates[0].synthetic_id == "synth-2"
    assert result.duplicates[0].reason == "duplicate_candidate_task_provider_output"


def test_dedupe_allows_distinct_output_hash_for_same_candidate() -> None:
    records = [_record("synth-1"), _record("synth-2", output_hash="different")]

    result = dedupe_records(records)

    assert len(result.records) == 2
    assert result.duplicates == []
