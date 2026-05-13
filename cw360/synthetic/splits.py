from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from cw360.synthetic.schemas import StructuredSyntheticRecord

SyntheticSplit = Literal["train", "val", "test"]


@dataclass(frozen=True, slots=True)
class SplitFractions:
    train: float = 0.98
    val: float = 0.01
    test: float = 0.01

    def validate(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1.0e-9:
            raise ValueError("split fractions must sum to 1.0")
        if self.train < 0 or self.val < 0 or self.test < 0:
            raise ValueError("split fractions must be non-negative")


def split_key(record: StructuredSyntheticRecord) -> str:
    candidate_key = record.source_candidate_id or record.candidate_id
    if candidate_key:
        return f"candidate:{candidate_key}"
    return f"synthetic:{record.synthetic_id}"


def assign_split(
    record: StructuredSyntheticRecord,
    *,
    fractions: SplitFractions | None = None,
) -> SyntheticSplit:
    active = fractions or SplitFractions()
    active.validate()
    bucket = stable_bucket(split_key(record))
    if bucket < active.train:
        return "train"
    if bucket < active.train + active.val:
        return "val"
    return "test"


def split_records(
    records: Iterable[StructuredSyntheticRecord],
    *,
    fractions: SplitFractions | None = None,
) -> dict[SyntheticSplit, list[StructuredSyntheticRecord]]:
    active = fractions or SplitFractions()
    active.validate()
    output: dict[SyntheticSplit, list[StructuredSyntheticRecord]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    key_to_split: dict[str, SyntheticSplit] = {}
    for record in records:
        key = split_key(record)
        split = key_to_split.setdefault(key, assign_split(record, fractions=active))
        output[split].append(record)
    return output


def stable_bucket(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return int(digest, 16) / 0xFFFFFFFFFFFFFFFF
