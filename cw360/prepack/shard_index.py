from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

SplitName = Literal["train", "val", "test"]
SPLITS: tuple[SplitName, ...] = ("train", "val", "test")


def stable_bucket(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return int(digest, 16) / 0xFFFFFFFFFFFFFFFF


def assign_split_for_key(
    key: str,
    *,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> SplitName:
    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > 1.0e-9:
        raise ValueError("split fractions must sum to 1.0")
    bucket = stable_bucket(key)
    if bucket < train_fraction:
        return "train"
    if bucket < train_fraction + val_fraction:
        return "val"
    return "test"


def grouped_shard_paths(
    manifest: Mapping[str, Any],
    root: str | Path,
) -> dict[SplitName, list[Path]]:
    base = Path(root)
    grouped: dict[SplitName, list[Path]] = {"train": [], "val": [], "test": []}
    file_list = manifest.get("shard_file_list")
    if not isinstance(file_list, dict):
        raise ValueError("manifest missing shard_file_list")
    for split in SPLITS:
        rows = file_list.get(split, [])
        if not isinstance(rows, list):
            raise ValueError(f"manifest shard_file_list.{split} must be a list")
        grouped[split] = [base / str(row) for row in rows]
    return grouped
