from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TextIO

from cw360.teacher.candidate import TeacherCandidate
from cw360.teacher.candidate_manifest import CandidateManifest, write_candidate_manifest


class CandidateShardWriter:
    def __init__(self, output_dir: str | Path, *, gzip_output: bool = False) -> None:
        self.output_dir = Path(output_dir)
        self.gzip_output = gzip_output

    def write_candidates(
        self,
        candidates: Iterable[TeacherCandidate],
        *,
        config: dict[str, Any] | None = None,
        write_manifest: bool = True,
    ) -> CandidateManifest:
        grouped: dict[tuple[str, str], list[TeacherCandidate]] = defaultdict(list)
        for candidate in candidates:
            grouped[(candidate.task_type, candidate.source)].append(candidate)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        shards: list[dict[str, Any]] = []
        task_counts: Counter[str] = Counter()
        source_counts: Counter[str] = Counter()
        total = 0

        for (task_type, source), rows in sorted(grouped.items()):
            relative = self._relative_shard_path(task_type, source)
            path = self.output_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._open_text(path) as handle:
                for candidate in rows:
                    handle.write(json.dumps(candidate.to_dict(), sort_keys=True) + "\n")
            count = len(rows)
            total += count
            task_counts[task_type] += count
            source_counts[source] += count
            shards.append(
                {
                    "path": relative.as_posix(),
                    "task_type": task_type,
                    "source": source,
                    "format": "jsonl.gz" if self.gzip_output else "jsonl",
                    "count": count,
                }
            )

        manifest = CandidateManifest(
            output_dir=str(self.output_dir),
            total_candidates=total,
            shards=shards,
            task_counts=dict(task_counts),
            source_counts=dict(source_counts),
            config=dict(config or {}),
            remote_upload={"enabled": False, "status": "not_configured"},
        )
        if write_manifest:
            write_candidate_manifest(manifest, self.output_dir / "candidate_manifest.json")
        return manifest

    def _relative_shard_path(self, task_type: str, source: str) -> Path:
        suffix = ".jsonl.gz" if self.gzip_output else ".jsonl"
        return Path(task_type) / f"{source}_candidates{suffix}"

    def _open_text(self, path: Path) -> TextIO:
        if self.gzip_output:
            return gzip.open(path, "wt", encoding="utf-8")
        return path.open("w", encoding="utf-8")


def read_candidate_shard(path: str | Path) -> list[TeacherCandidate]:
    shard_path = Path(path)
    opener = gzip.open if shard_path.suffix == ".gz" else open
    with opener(shard_path, "rt", encoding="utf-8") as handle:
        return [TeacherCandidate.from_dict(json.loads(line)) for line in handle if line.strip()]
