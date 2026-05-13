from __future__ import annotations

import pytest

from cw360.synthetic.schemas import StructuredSyntheticRecord
from cw360.synthetic.splits import SplitFractions, split_key, split_records
from tests.test_synthetic_schema import sample_payload


def _record(identifier: str, source_candidate_id: str) -> StructuredSyntheticRecord:
    return StructuredSyntheticRecord.from_dict(
        sample_payload(
            synthetic_id=identifier,
            source_candidate_id=source_candidate_id,
            output_hash=f"output-{identifier}",
        )
    )


def test_splits_keep_same_candidate_key_together() -> None:
    records = [_record("synth-1", "candidate-a"), _record("synth-2", "candidate-a")]

    splits = split_records(records, fractions=SplitFractions(train=0.5, val=0.25, test=0.25))

    non_empty = [split for split, rows in splits.items() if rows]
    assert len(non_empty) == 1
    assert split_key(records[0]) == "candidate:candidate-a"


def test_splits_reject_bad_fraction_sum() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        split_records([_record("synth-1", "candidate-a")], fractions=SplitFractions(1, 1, 1))
