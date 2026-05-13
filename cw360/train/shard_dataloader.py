from __future__ import annotations

import random
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cw360.checkpoint.data_cursor import TPUDataCursor
from cw360.checkpoint.metadata import DataSnapshot
from cw360.prepack.manifest import hash_payload, read_json
from cw360.prepack.shard_index import SplitName, grouped_shard_paths
from cw360.train.trainer import TrainingBatch
from cw360.train.validate import validate_static_shape_rules


@dataclass(frozen=True, slots=True)
class PrepackedShardInfo:
    path: Path
    shard_index: int
    num_sequences: int


class PrepackedShardDataLoader:
    def __init__(
        self,
        input_dir: str | Path,
        *,
        split: SplitName = "train",
        batch_size: int,
        drop_last: bool = True,
        static_shapes: bool = True,
        shard_order_seed: int = 0,
        cursor: TPUDataCursor | None = None,
        num_devices: int | None = None,
        resume_policy: str = "resume_exact_cursor",
    ) -> None:
        self.input_dir = Path(input_dir)
        self.split: SplitName = split
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.static_shapes = bool(static_shapes)
        self.shard_order_seed = int(shard_order_seed)
        self.cursor = cursor
        self.num_devices = num_devices
        self.resume_policy = resume_policy

        self.manifest = read_json(self.input_dir / "manifest.json")
        self.max_seq_len = int(self.manifest["max_seq_len"])
        self.seq_len_plus_one = int(self.manifest["seq_len_plus_one"])
        validate_static_shape_rules(
            max_seq_len=self.max_seq_len,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            static_shapes=self.static_shapes,
        )
        if self.seq_len_plus_one != self.max_seq_len + 1:
            raise ValueError("prepacked manifest seq_len_plus_one must equal max_seq_len + 1")

        self.shard_paths = grouped_shard_paths(self.manifest, self.input_dir)[split]
        if not self.shard_paths:
            raise ValueError(f"no {split} shards found in prepacked manifest")
        if self.shard_order_seed:
            rng = random.Random(self.shard_order_seed)
            rng.shuffle(self.shard_paths)
        if cursor is not None:
            self._validate_cursor(cursor)

    def __iter__(self) -> Iterator[TrainingBatch]:
        start_shard = 0 if self.cursor is None else self.cursor.shard_index
        start_offset = 0 if self.cursor is None else self.cursor.sequence_offset
        epoch = 0 if self.cursor is None else self.cursor.epoch
        sequences_seen = 0 if self.cursor is None else self.cursor.sequences_seen
        tokens_seen = 0 if self.cursor is None else self.cursor.tokens_seen
        global_step = 0 if self.cursor is None else self.cursor.global_step

        for shard_index, shard_path in enumerate(self.shard_paths[start_shard:], start=start_shard):
            loaded_at = time.perf_counter()
            shard = np.load(shard_path, mmap_mode="r")
            data_wait_time = time.perf_counter() - loaded_at
            self._validate_shard(shard_path, shard)
            offset = start_offset if shard_index == start_shard else 0
            usable_end = self._usable_end(int(shard.shape[0]), offset)
            while offset < usable_end:
                next_offset = offset + self.batch_size
                rows = shard[offset:next_offset]
                input_ids = torch.as_tensor(rows[:, :-1].astype(np.int64), dtype=torch.long)
                labels = torch.as_tensor(rows[:, 1:].astype(np.int64), dtype=torch.long)
                sequences_seen += self.batch_size
                tokens_seen += self.batch_size * self.max_seq_len
                next_shard_index = shard_index
                cursor_offset = next_offset
                consumed_in_current_shard = next_offset
                if next_offset >= int(shard.shape[0]):
                    next_shard_index = shard_index + 1
                    cursor_offset = 0
                    consumed_in_current_shard = 0
                yield TrainingBatch(
                    input_ids=input_ids,
                    labels=labels,
                    tpu_data_cursor=TPUDataCursor(
                        epoch=epoch,
                        global_step=global_step,
                        tokens_seen=tokens_seen,
                        sequences_seen=sequences_seen,
                        shard_order_seed=self.shard_order_seed,
                        shard_index=next_shard_index,
                        sequence_offset=cursor_offset,
                        consumed_sequences_in_current_shard=consumed_in_current_shard,
                        drop_last=self.drop_last,
                        batch_size_per_device=self.batch_size,
                        num_devices=self.num_devices,
                        resume_policy=self.resume_policy,
                    ),
                    data_wait_time=data_wait_time,
                )
                offset = next_offset

    def _usable_end(self, num_sequences: int, offset: int) -> int:
        remaining = num_sequences - offset
        full_batches = remaining // self.batch_size
        if full_batches <= 0:
            return offset
        return offset + full_batches * self.batch_size

    def _validate_shard(self, shard_path: Path, shard: np.ndarray) -> None:
        if shard.ndim != 2:
            raise ValueError(f"prepacked shard must be rank 2: {shard_path}")
        if int(shard.shape[1]) != self.seq_len_plus_one:
            raise ValueError(
                f"prepacked shard {shard_path} has sequence length {shard.shape[1]}, "
                f"expected {self.seq_len_plus_one}"
            )
        if shard.dtype != np.uint16:
            raise ValueError(f"prepacked shard dtype must be uint16: {shard_path}")

    def _validate_cursor(self, cursor: TPUDataCursor) -> None:
        if cursor.shard_order_seed != self.shard_order_seed:
            raise ValueError("cursor shard_order_seed does not match loader")
        if cursor.drop_last != self.drop_last:
            raise ValueError("cursor drop_last does not match loader")
        if cursor.batch_size_per_device != self.batch_size:
            raise ValueError("cursor batch_size_per_device does not match loader")
        if cursor.resume_policy != self.resume_policy:
            raise ValueError("cursor resume_policy does not match loader")
        if cursor.shard_index < 0 or cursor.shard_index > len(self.shard_paths):
            raise ValueError("cursor shard_index is out of range")
        if cursor.sequence_offset < 0:
            raise ValueError("cursor sequence_offset must be non-negative")


def data_snapshot_from_prepacked_manifest(input_dir: str | Path) -> DataSnapshot:
    root = Path(input_dir)
    manifest = read_json(root / "manifest.json")
    grouped_files = manifest["shard_file_list"]
    grouped_hashes = manifest["shard_hashes"]
    flat_files: list[str] = []
    flat_hashes: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for filename in grouped_files.get(split, []):
            name = str(filename)
            flat_files.append(name)
            flat_hashes[name] = str(grouped_hashes.get(split, {}).get(name, "missing"))
    shard_manifest_hash = hash_payload({"files": flat_files, "hashes": flat_hashes})
    return DataSnapshot(
        datasets_config_hash=str(manifest.get("data_snapshot_hash") or "prepacked-data"),
        mixture_config_hash=str(manifest.get("prepack_config_hash") or "prepack-config"),
        synthetic_repo_id=_none_or_str(manifest.get("synthetic_repo_id")),
        synthetic_shard_list=[],
        teacher_manifest_id=None,
        candidate_manifest_id=None,
        split_manifest_id=_none_or_str(manifest.get("split_manifest_id")),
        tokenizer_id=str(manifest["tokenizer_id"]),
        tokenizer_config_hash=_none_or_str(manifest.get("tokenizer_config_hash")),
        prepacked_dataset_name=str(manifest["dataset_name"]),
        prepacked_dataset_version=str(manifest["dataset_version"]),
        prepacked_manifest_hash=str(manifest["manifest_hash"]),
        shard_manifest_hash=shard_manifest_hash,
        shard_file_list=flat_files,
        shard_hashes=flat_hashes,
        seq_len_plus_one=int(manifest["seq_len_plus_one"]),
        token_dtype=str(manifest["dtype"]),
    )


def validate_prepacked_manifest_for_training(manifest: Mapping[str, Any]) -> None:
    if int(manifest["seq_len_plus_one"]) != int(manifest["max_seq_len"]) + 1:
        raise ValueError("prepacked manifest must use max_seq_len + 1 rows")
    if str(manifest["dtype"]) != "uint16":
        raise ValueError("prepacked manifest dtype must be uint16")
    if str(manifest["shard_format"]) != "npy":
        raise ValueError("prepacked manifest shard_format must be npy")


def _none_or_str(value: object) -> str | None:
    return None if value is None else str(value)
