from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CandidateManifest:
    output_dir: str
    total_candidates: int
    shards: list[dict[str, Any]]
    task_counts: dict[str, int]
    source_counts: dict[str, int]
    config: dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    remote_upload: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_type": "teacher_candidate_manifest",
            "generated_at": self.generated_at,
            "output_dir": self.output_dir,
            "total_candidates": self.total_candidates,
            "shards": self.shards,
            "task_counts": self.task_counts,
            "source_counts": self.source_counts,
            "config": self.config,
            "remote_upload": self.remote_upload,
        }


def write_candidate_manifest(manifest: CandidateManifest, path: str | Path) -> Path:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest_path


def load_candidate_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("manifest_type") != "teacher_candidate_manifest":
        raise ValueError(f"{path} is not a teacher candidate manifest")
    return payload
