from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from cw360.data.schemas import TrainingExample

SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    split: SplitName
    key: str
    bucket: float


def split_key(example: TrainingExample) -> str:
    candidate_id = example.metadata.get("candidate_id") or example.metadata.get(
        "source_candidate_id"
    )
    if candidate_id:
        return f"candidate:{candidate_id}"

    stable_id = example.metadata.get("id") or example.metadata.get("record_id")
    if stable_id:
        return f"{example.source}:{stable_id}"

    digest = hashlib.sha256(example.text.encode("utf-8")).hexdigest()
    return f"{example.source}:{example.example_type}:text:{digest}"


def assign_split(
    example: TrainingExample,
    *,
    train_pct: float = 98.0,
    validation_pct: float = 1.0,
    test_pct: float = 1.0,
) -> SplitAssignment:
    _validate_percentages(train_pct, validation_pct, test_pct)
    key = split_key(example)
    bucket = _bucket_for_key(key)
    if bucket < test_pct:
        split: SplitName = "test"
    elif bucket < test_pct + validation_pct:
        split = "validation"
    else:
        split = "train"
    return SplitAssignment(split=split, key=key, bucket=bucket)


def split_examples(
    examples: Iterable[TrainingExample],
    *,
    train_pct: float = 98.0,
    validation_pct: float = 1.0,
    test_pct: float = 1.0,
) -> dict[SplitName, list[TrainingExample]]:
    output: dict[SplitName, list[TrainingExample]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    key_to_split: dict[str, SplitName] = {}

    for example in examples:
        assignment = assign_split(
            example,
            train_pct=train_pct,
            validation_pct=validation_pct,
            test_pct=test_pct,
        )
        split = key_to_split.setdefault(assignment.key, assignment.split)
        output[split].append(example)

    return output


def write_split_manifest(
    examples: Iterable[TrainingExample],
    path: str | Path,
    *,
    train_pct: float = 98.0,
    validation_pct: float = 1.0,
    test_pct: float = 1.0,
) -> dict[str, object]:
    counts: dict[str, int] = defaultdict(int)
    keys: dict[str, str] = {}
    assignments: list[dict[str, object]] = []

    for index, example in enumerate(examples):
        assignment = assign_split(
            example,
            train_pct=train_pct,
            validation_pct=validation_pct,
            test_pct=test_pct,
        )
        counts[assignment.split] += 1
        keys[assignment.key] = assignment.split
        assignments.append(
            {
                "index": index,
                "split": assignment.split,
                "key": assignment.key,
                "bucket": assignment.bucket,
                "source": example.source,
                "example_type": example.example_type,
            }
        )

    manifest: dict[str, object] = {
        "split_percentages": {
            "train": train_pct,
            "validation": validation_pct,
            "test": test_pct,
        },
        "counts": dict(counts),
        "unique_split_keys": len(keys),
        "assignments": assignments,
    }
    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _validate_percentages(train_pct: float, validation_pct: float, test_pct: float) -> None:
    if train_pct < 0 or validation_pct < 0 or test_pct < 0:
        raise ValueError("split percentages must be non-negative")
    total = train_pct + validation_pct + test_pct
    if abs(total - 100.0) > 1e-9:
        raise ValueError("split percentages must sum to 100")


def _bucket_for_key(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return (int(digest, 16) / 0xFFFFFFFFFFFFFFFF) * 100.0
