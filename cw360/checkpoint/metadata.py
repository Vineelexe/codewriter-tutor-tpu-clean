from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cw360.checkpoint.atomic import atomic_write_json
from cw360.checkpoint.data_cursor import Backend, TPUDataCursor


@dataclass(frozen=True, slots=True)
class DataSnapshot:
    datasets_config_hash: str
    mixture_config_hash: str
    synthetic_repo_id: str | None
    synthetic_shard_list: list[str]
    teacher_manifest_id: str | None
    candidate_manifest_id: str | None
    split_manifest_id: str | None
    tokenizer_id: str
    tokenizer_config_hash: str | None
    prepacked_dataset_name: str
    prepacked_dataset_version: str
    prepacked_manifest_hash: str
    shard_manifest_hash: str
    shard_file_list: list[str]
    shard_hashes: dict[str, str]
    seq_len_plus_one: int
    token_dtype: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataSnapshot":
        return cls(
            datasets_config_hash=str(data["datasets_config_hash"]),
            mixture_config_hash=str(data["mixture_config_hash"]),
            synthetic_repo_id=none_or_str(data.get("synthetic_repo_id")),
            synthetic_shard_list=[str(item) for item in data["synthetic_shard_list"]],
            teacher_manifest_id=none_or_str(data.get("teacher_manifest_id")),
            candidate_manifest_id=none_or_str(data.get("candidate_manifest_id")),
            split_manifest_id=none_or_str(data.get("split_manifest_id")),
            tokenizer_id=str(data["tokenizer_id"]),
            tokenizer_config_hash=none_or_str(data.get("tokenizer_config_hash")),
            prepacked_dataset_name=str(data["prepacked_dataset_name"]),
            prepacked_dataset_version=str(data["prepacked_dataset_version"]),
            prepacked_manifest_hash=str(data["prepacked_manifest_hash"]),
            shard_manifest_hash=str(data["shard_manifest_hash"]),
            shard_file_list=[str(item) for item in data["shard_file_list"]],
            shard_hashes={str(key): str(value) for key, value in data["shard_hashes"].items()},
            seq_len_plus_one=int(data["seq_len_plus_one"]),
            token_dtype=str(data["token_dtype"]),
        )


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    model_name: str
    model_size_label: str
    run_id: str
    step: int
    train_loss: float
    val_loss: float | None
    tokens_seen: int
    sequences_seen: int
    trained_by: str
    date: str
    platform: str
    training_stage: str
    previous_checkpoint: str | None
    dataset_mixture: dict[str, float]
    context_length: int
    optimizer: str
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int
    precision: str
    backend: Backend
    git_commit: str | None
    notes: str | None
    data_snapshot: DataSnapshot
    tpu_data_cursor: TPUDataCursor | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["data_snapshot"] = self.data_snapshot.to_dict()
        data["tpu_data_cursor"] = (
            None if self.tpu_data_cursor is None else self.tpu_data_cursor.to_dict()
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointMetadata":
        cursor_data = data.get("tpu_data_cursor")
        return cls(
            model_name=str(data["model_name"]),
            model_size_label=str(data["model_size_label"]),
            run_id=str(data["run_id"]),
            step=int(data["step"]),
            train_loss=float(data["train_loss"]),
            val_loss=None if data.get("val_loss") is None else float(data["val_loss"]),
            tokens_seen=int(data["tokens_seen"]),
            sequences_seen=int(data["sequences_seen"]),
            trained_by=str(data["trained_by"]),
            date=str(data["date"]),
            platform=str(data["platform"]),
            training_stage=str(data["training_stage"]),
            previous_checkpoint=none_or_str(data.get("previous_checkpoint")),
            dataset_mixture={
                str(key): float(value) for key, value in data["dataset_mixture"].items()
            },
            context_length=int(data["context_length"]),
            optimizer=str(data["optimizer"]),
            learning_rate=float(data["learning_rate"]),
            batch_size=int(data["batch_size"]),
            gradient_accumulation_steps=int(data["gradient_accumulation_steps"]),
            precision=str(data["precision"]),
            backend=str(data["backend"]),  # type: ignore[arg-type]
            git_commit=none_or_str(data.get("git_commit")),
            notes=none_or_str(data.get("notes")),
            data_snapshot=DataSnapshot.from_dict(data["data_snapshot"]),
            tpu_data_cursor=None if cursor_data is None else TPUDataCursor.from_dict(cursor_data),
        )


def none_or_str(value: object) -> str | None:
    return None if value is None else str(value)


def metadata_path_for_checkpoint(checkpoint_path: str | Path) -> Path:
    return Path(checkpoint_path).with_suffix(".json")


def write_metadata_json(metadata: CheckpointMetadata, checkpoint_path: str | Path) -> Path:
    metadata_path = metadata_path_for_checkpoint(checkpoint_path)
    atomic_write_json(metadata.to_dict(), metadata_path)
    return metadata_path


def read_metadata_json(path: str | Path) -> CheckpointMetadata:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"metadata file must contain a JSON object: {path}")
    return CheckpointMetadata.from_dict(data)


def get_git_commit(cwd: str | Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=None if cwd is None else str(cwd),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    if result.returncode != 0 or not commit:
        return None
    return commit
