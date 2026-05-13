from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cw360.prepack.manifest import read_json, sha256_file
from cw360.prepack.mixture_audit import audit_mixture
from cw360.prepack.shard_index import SPLITS
from cw360.prepack.writer import LOSS_WEIGHT_HANDLING_POLICY


class PrepackVerificationError(AssertionError):
    pass


@dataclass(frozen=True, slots=True)
class VerificationReport:
    input_dir: Path
    shard_count: int
    total_sequences: int
    total_tokens_for_training: int


def verify_prepacked_dataset(input_dir: str | Path) -> VerificationReport:
    root = Path(input_dir)
    manifest_path = root / "manifest.json"
    kaggle_metadata_path = root / "dataset-metadata.json"
    if not manifest_path.exists():
        raise PrepackVerificationError("manifest.json is missing")
    if not kaggle_metadata_path.exists():
        raise PrepackVerificationError("dataset-metadata.json is missing")

    manifest = read_json(manifest_path)
    _assert_split_fractions(manifest)
    _assert_split_directories(root)
    if manifest.get("loss_weight_handling_policy") != LOSS_WEIGHT_HANDLING_POLICY:
        raise PrepackVerificationError("loss_weight policy must be prepack sampling/oversampling")

    target_mixture = _dict(manifest, "target_mixture")
    tolerance = float(manifest.get("validation", {}).get("shard_mixture_tolerance_abs", 0.03))
    if "validation" not in manifest:
        tolerance = 0.03

    global_counts: dict[str, int] = {}
    shard_rows = manifest.get("shards")
    if not isinstance(shard_rows, list):
        raise PrepackVerificationError("manifest missing shards list")
    for row in shard_rows:
        if not isinstance(row, dict):
            raise PrepackVerificationError("shard row must be an object")
        shard_path = root / str(row["shard_filename"])
        metadata_path = shard_path.with_suffix(".json")
        if not metadata_path.exists():
            raise PrepackVerificationError(f"shard metadata is missing: {metadata_path}")
        metadata = read_json(metadata_path)
        _verify_shard_array(shard_path, metadata, manifest)
        if int(metadata.get("atomic_boundary_violations", 0)) != 0:
            raise PrepackVerificationError("atomic examples crossed sequence boundary")
        if metadata.get("loss_weight_handling_summary") != LOSS_WEIGHT_HANDLING_POLICY:
            raise PrepackVerificationError("shard loss_weight policy is invalid")
        source_counts = _dict(metadata, "source_counts")
        for key, value in source_counts.items():
            global_counts[key] = global_counts.get(key, 0) + int(value)
        audit_mixture(
            actual_counts=source_counts,
            target_weights=target_mixture,
            tolerance_abs=tolerance,
            label=str(metadata["shard_filename"]),
        )

    audit_mixture(
        actual_counts=global_counts,
        target_weights=target_mixture,
        tolerance_abs=tolerance,
        label="global",
    )
    return VerificationReport(
        input_dir=root,
        shard_count=int(manifest["shard_count"]),
        total_sequences=int(manifest["total_sequences"]),
        total_tokens_for_training=int(manifest["total_tokens_for_training"]),
    )


def _verify_shard_array(
    shard_path: Path,
    metadata: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    if not shard_path.exists():
        raise PrepackVerificationError(f"shard is missing: {shard_path}")
    if sha256_file(shard_path) != metadata.get("sha256"):
        raise PrepackVerificationError(f"sha256 mismatch for {shard_path}")
    array = np.load(shard_path, mmap_mode="r")
    expected_seq_len = int(manifest["seq_len_plus_one"])
    if array.ndim != 2 or array.shape[1] != expected_seq_len:
        raise PrepackVerificationError(
            f"{shard_path} shape must be [N, {expected_seq_len}], got {array.shape}"
        )
    if array.shape[0] <= 0:
        raise PrepackVerificationError(f"{shard_path} is empty")
    if array.dtype != np.dtype("uint16"):
        raise PrepackVerificationError(f"{shard_path} dtype must be uint16")
    if int(array.max()) > 65_535:
        raise PrepackVerificationError(f"{shard_path} contains token overflow")
    if list(array.shape) != [int(metadata["num_sequences"]), expected_seq_len]:
        raise PrepackVerificationError(f"{shard_path} shape does not match metadata")


def _assert_split_fractions(manifest: dict[str, Any]) -> None:
    fractions = _dict(manifest, "split_fractions")
    total = sum(float(fractions.get(split, 0.0)) for split in SPLITS)
    if abs(total - 1.0) > 1.0e-9:
        raise PrepackVerificationError("split fractions must sum to 1.0")


def _assert_split_directories(root: Path) -> None:
    for split in ("train", "val"):
        if not (root / split).is_dir():
            raise PrepackVerificationError(f"{split}/ shard directory is missing")


def _dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise PrepackVerificationError(f"{key} must be an object")
    return value
