from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cw360.teacher.structured_record import utc_now_iso


@dataclass(slots=True)
class TeacherRunManifest:
    run_id: str
    provider: str
    model: str
    output_dir: str
    started_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    completed_candidates: dict[str, str] = field(default_factory=dict)
    failed_candidates: dict[str, str] = field(default_factory=dict)
    written_shards: list[dict[str, Any]] = field(default_factory=list)
    total_requests: int = 0
    dry_run: bool = False

    def mark_completed(self, candidate_id: str, synthetic_id: str) -> None:
        self.completed_candidates[candidate_id] = synthetic_id
        self.failed_candidates.pop(candidate_id, None)
        self.total_requests += 1
        self.updated_at = utc_now_iso()

    def mark_failed(self, candidate_id: str, reason: str) -> None:
        self.failed_candidates[candidate_id] = reason
        self.total_requests += 1
        self.updated_at = utc_now_iso()

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "provider": self.provider,
            "model": self.model,
            "output_dir": self.output_dir,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "completed_candidates": dict(self.completed_candidates),
            "failed_candidates": dict(self.failed_candidates),
            "written_shards": list(self.written_shards),
            "total_requests": self.total_requests,
            "dry_run": self.dry_run,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TeacherRunManifest:
        return cls(
            run_id=str(payload["run_id"]),
            provider=str(payload["provider"]),
            model=str(payload["model"]),
            output_dir=str(payload["output_dir"]),
            started_at=str(payload.get("started_at") or utc_now_iso()),
            updated_at=str(payload.get("updated_at") or utc_now_iso()),
            completed_candidates=dict(payload.get("completed_candidates") or {}),
            failed_candidates=dict(payload.get("failed_candidates") or {}),
            written_shards=list(payload.get("written_shards") or []),
            total_requests=int(payload.get("total_requests", 0)),
            dry_run=bool(payload.get("dry_run", False)),
        )


def manifest_path(output_dir: str | Path, run_id: str) -> Path:
    return Path(output_dir) / f"{run_id}_manifest.json"


def load_manifest(path: str | Path) -> TeacherRunManifest | None:
    manifest_file = Path(path)
    if not manifest_file.exists():
        return None
    with manifest_file.open("r", encoding="utf-8") as handle:
        return TeacherRunManifest.from_dict(json.load(handle))


def save_manifest(manifest: TeacherRunManifest, path: str | Path) -> None:
    manifest_file = Path(path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    temp_path = manifest_file.with_suffix(manifest_file.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temp_path, manifest_file)
