from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from cw360.teacher.candidate import TeacherCandidate
from cw360.teacher.candidate_shards import read_candidate_shard
from cw360.teacher.prompts import PromptBundle, build_teacher_prompt
from cw360.teacher.structured_record import stable_hash


@dataclass(frozen=True, slots=True)
class TeacherRequest:
    candidate: TeacherCandidate
    prompt: str
    system_prompt: str
    request_hash: str


def build_request(candidate: TeacherCandidate) -> TeacherRequest:
    bundle: PromptBundle = build_teacher_prompt(candidate)
    request_hash = stable_hash(
        json.dumps(
            {
                "candidate_id": candidate.candidate_id,
                "input_hash": candidate.input_hash,
                "task_type": candidate.task_type,
                "prompt": bundle.prompt,
            },
            sort_keys=True,
        )
    )
    return TeacherRequest(
        candidate=candidate,
        prompt=bundle.prompt,
        system_prompt=bundle.system_prompt,
        request_hash=request_hash,
    )


def load_candidate_inputs(paths: list[str | Path]) -> list[TeacherCandidate]:
    candidates: list[TeacherCandidate] = []
    for raw_path in paths:
        path = Path(raw_path)
        candidate_files = _resolve_candidate_files(path)
        for file_path in candidate_files:
            candidates.extend(read_candidate_shard(file_path))
    return candidates


def build_request_queue(
    candidates: list[TeacherCandidate],
    *,
    completed_candidate_ids: set[str] | None = None,
) -> list[TeacherRequest]:
    completed = completed_candidate_ids or set()
    requests: list[TeacherRequest] = []
    seen_hashes: set[str] = set()
    for candidate in candidates:
        if candidate.candidate_id in completed:
            continue
        request = build_request(candidate)
        if request.request_hash in seen_hashes:
            continue
        seen_hashes.add(request.request_hash)
        requests.append(request)
    return requests


def _resolve_candidate_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    search_dir = path
    if not path.exists() and path.name == "candidates.jsonl":
        search_dir = path.parent
    if search_dir.is_dir():
        files = list(search_dir.rglob("*_candidates.jsonl")) + list(
            search_dir.rglob("*_candidates.jsonl.gz")
        )
        return sorted(files)
    raise FileNotFoundError(f"candidate shard path does not exist: {path}")
