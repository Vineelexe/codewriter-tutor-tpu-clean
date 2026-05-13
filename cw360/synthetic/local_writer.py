from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cw360.synthetic.schemas import StructuredSyntheticRecord


def write_records_jsonl(
    records: list[StructuredSyntheticRecord],
    path: str | Path,
) -> None:
    rows = [record.to_dict() for record in records]
    write_rows_jsonl(rows, path)


def write_rows_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def read_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path} must contain JSON objects")
            rows.append(payload)
    return rows
