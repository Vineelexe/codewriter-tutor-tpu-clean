from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cw360.teacher.structured_record import StructuredTrainingRecord


@dataclass(frozen=True, slots=True)
class WrittenShard:
    path: str
    count: int
    format: str = "jsonl"


class StructuredShardWriter:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        shard_size: int = 1000,
        starting_index: int = 0,
    ) -> None:
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        if starting_index < 0:
            raise ValueError("starting_index must be non-negative")
        self.output_dir = Path(output_dir)
        self.shard_size = shard_size
        self._buffer: list[StructuredTrainingRecord] = []
        self._shard_index = starting_index
        self.written: list[WrittenShard] = []

    def write(self, record: StructuredTrainingRecord) -> list[WrittenShard]:
        self._buffer.append(record)
        if len(self._buffer) >= self.shard_size:
            return [self.flush()]
        return []

    def flush(self) -> WrittenShard:
        if not self._buffer:
            return WrittenShard(path="", count=0)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"structured_{self._shard_index:05d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in self._buffer:
                handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
        shard = WrittenShard(path=path.name, count=len(self._buffer))
        self.written.append(shard)
        self._buffer = []
        self._shard_index += 1
        return shard

    def written_manifest_rows(self) -> list[dict[str, Any]]:
        return [
            {"path": shard.path, "count": shard.count, "format": shard.format}
            for shard in self.written
            if shard.count > 0
        ]


def read_structured_shard(path: str | Path) -> list[StructuredTrainingRecord]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [
            StructuredTrainingRecord.from_dict(json.loads(line))
            for line in handle
            if line.strip()
        ]
