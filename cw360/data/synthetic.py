from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from cw360.data.extractors import extract_synthetic_jsonl
from cw360.data.schemas import ExtractedRecord


def iter_synthetic_jsonl(path: str | Path) -> Iterator[ExtractedRecord]:
    jsonl_path = Path(path)
    try:
        handle = jsonl_path.open("r", encoding="utf-8")
    except OSError as exc:
        raise FileNotFoundError(f"could not read synthetic JSONL {jsonl_path}: {exc}") from exc

    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row: Any = json.loads(line)
            except json.JSONDecodeError as exc:
                yield ExtractedRecord(
                    None,
                    error=f"line {line_number}: malformed JSON: {exc.msg}",
                    line_number=line_number,
                )
                continue
            if not isinstance(row, dict):
                yield ExtractedRecord(
                    None, error=f"line {line_number}: expected JSON object", line_number=line_number
                )
                continue
            try:
                example = extract_synthetic_jsonl(row)
            except (TypeError, ValueError) as exc:
                yield ExtractedRecord(
                    None,
                    error=f"line {line_number}: malformed example: {exc}",
                    line_number=line_number,
                )
                continue
            if not example.text.strip():
                yield ExtractedRecord(
                    None, error=f"line {line_number}: missing text", line_number=line_number
                )
                continue
            yield ExtractedRecord(example, line_number=line_number)
