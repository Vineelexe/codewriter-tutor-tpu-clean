from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class TeacherResponseCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._items: dict[str, dict[str, Any]] = {}
        self._load()

    def get(self, request_hash: str) -> dict[str, Any] | None:
        value = self._items.get(request_hash)
        return dict(value) if value is not None else None

    def set(self, request_hash: str, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {"request_hash": request_hash, "payload": payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        self._items[request_hash] = dict(payload)

    def __contains__(self, request_hash: str) -> bool:
        return request_hash in self._items

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                self._items[str(row["request_hash"])] = dict(row["payload"])
