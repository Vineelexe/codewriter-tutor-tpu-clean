from __future__ import annotations

import json
from pathlib import Path

from cw360.data.schemas import TrainingExample
from cw360.data.splits import assign_split, split_examples, split_key, write_split_manifest


def test_synthetic_split_key_uses_candidate_id_to_prevent_leakage() -> None:
    first = TrainingExample(
        text="prompt one",
        source="synthetic",
        example_type="debugging",
        metadata={"candidate_id": "cand-123"},
    )
    second = TrainingExample(
        text="different prompt from same candidate",
        source="synthetic",
        example_type="tests",
        metadata={"candidate_id": "cand-123"},
    )

    assert split_key(first) == "candidate:cand-123"
    assert assign_split(first).split == assign_split(second).split

    splits = split_examples([first, second])
    populated = [name for name, items in splits.items() if items]
    assert len(populated) == 1
    assert len(splits[populated[0]]) == 2


def test_source_candidate_id_is_also_grouped() -> None:
    example = TrainingExample(
        text="structured synthetic",
        source="synthetic",
        example_type="comments",
        metadata={"source_candidate_id": "src-7"},
    )

    assert split_key(example) == "candidate:src-7"


def test_write_split_manifest_records_assignments() -> None:
    output_dir = Path(".pytest-tmp-phase7")
    output_dir.mkdir(exist_ok=True)
    path = output_dir / "split_manifest.json"
    examples = [
        TrainingExample(text="a", source="unit", example_type="python_code", metadata={"id": "1"}),
        TrainingExample(text="b", source="unit", example_type="python_code", metadata={"id": "2"}),
    ]

    manifest = write_split_manifest(examples, path)
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert loaded == manifest
    assert loaded["split_percentages"] == {"train": 98.0, "validation": 1.0, "test": 1.0}
    assert loaded["unique_split_keys"] == 2
    assert len(loaded["assignments"]) == 2
