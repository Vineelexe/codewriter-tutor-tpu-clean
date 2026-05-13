from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cw360.prepack.manifest import read_json
from cw360.prepack.shard_index import SplitName, grouped_shard_paths


@dataclass(frozen=True, slots=True)
class PrepackedDatasetCursor:
    epoch: int
    shard_order_seed: int
    shard_index: int
    sequence_offset: int
    consumed_sequences: int
    tokens_seen: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PrepackedDatasetCursor:
        return cls(
            epoch=int(data["epoch"]),
            shard_order_seed=int(data["shard_order_seed"]),
            shard_index=int(data["shard_index"]),
            sequence_offset=int(data["sequence_offset"]),
            consumed_sequences=int(data["consumed_sequences"]),
            tokens_seen=int(data["tokens_seen"]),
        )


@dataclass(frozen=True, slots=True)
class PrepackedBatch:
    input_ids: np.ndarray
    labels: np.ndarray
    cursor: PrepackedDatasetCursor


class PrepackedShardReader:
    def __init__(
        self,
        input_dir: str | Path,
        *,
        split: SplitName = "train",
        shard_order_seed: int = 0,
        mmap: bool = True,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.split = split
        self.shard_order_seed = shard_order_seed
        self.mmap = mmap
        self.manifest = read_json(self.input_dir / "manifest.json")
        self.max_seq_len = int(self.manifest["max_seq_len"])
        self.seq_len_plus_one = int(self.manifest["seq_len_plus_one"])
        self.shard_paths = grouped_shard_paths(self.manifest, self.input_dir)[split]
        if shard_order_seed:
            rng = random.Random(shard_order_seed)
            rng.shuffle(self.shard_paths)

    def iter_sequences(
        self,
        *,
        cursor: PrepackedDatasetCursor | None = None,
    ) -> Iterator[tuple[np.ndarray, PrepackedDatasetCursor]]:
        start_shard = 0 if cursor is None else cursor.shard_index
        start_offset = 0 if cursor is None else cursor.sequence_offset
        consumed = 0 if cursor is None else cursor.consumed_sequences
        for shard_index, shard_path in enumerate(self.shard_paths[start_shard:], start=start_shard):
            array = np.load(shard_path, mmap_mode="r" if self.mmap else None)
            if array.ndim != 2 or array.shape[1] != self.seq_len_plus_one:
                raise ValueError(f"invalid shard shape for {shard_path}: {array.shape}")
            offset = start_offset if shard_index == start_shard else 0
            for sequence_offset in range(offset, array.shape[0]):
                consumed += 1
                next_offset = sequence_offset + 1
                next_shard_index = shard_index
                if next_offset >= array.shape[0]:
                    next_shard_index = shard_index + 1
                    next_offset = 0
                yield array[sequence_offset], PrepackedDatasetCursor(
                    epoch=0,
                    shard_order_seed=self.shard_order_seed,
                    shard_index=next_shard_index,
                    sequence_offset=next_offset,
                    consumed_sequences=consumed,
                    tokens_seen=consumed * self.max_seq_len,
                )

    def iter_batches(
        self,
        *,
        batch_size: int,
        cursor: PrepackedDatasetCursor | None = None,
        drop_last: bool = True,
    ) -> Iterator[PrepackedBatch]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        sequences: list[np.ndarray] = []
        last_cursor: PrepackedDatasetCursor | None = cursor
        for sequence, sequence_cursor in self.iter_sequences(cursor=cursor):
            sequences.append(np.asarray(sequence, dtype=np.uint16))
            last_cursor = sequence_cursor
            if len(sequences) == batch_size:
                stacked = np.stack(sequences)
                yield PrepackedBatch(
                    input_ids=stacked[:, :-1],
                    labels=stacked[:, 1:],
                    cursor=sequence_cursor,
                )
                sequences = []
        if sequences and not drop_last and last_cursor is not None:
            stacked = np.stack(sequences)
            yield PrepackedBatch(
                input_ids=stacked[:, :-1],
                labels=stacked[:, 1:],
                cursor=last_cursor,
            )
