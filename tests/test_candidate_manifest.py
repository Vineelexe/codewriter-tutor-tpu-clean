from __future__ import annotations

import pytest

from cw360.teacher.candidate_manifest import (
    CandidateManifest,
    load_candidate_manifest,
    write_candidate_manifest,
)


def test_candidate_manifest_round_trip(tmp_path) -> None:
    manifest = CandidateManifest(
        output_dir=str(tmp_path),
        total_candidates=1,
        shards=[
            {
                "path": "debugging/debugbench_candidates.jsonl",
                "task_type": "debugging",
                "source": "debugbench",
                "format": "jsonl",
                "count": 1,
            }
        ],
        task_counts={"debugging": 1},
        source_counts={"debugbench": 1},
        config={"min_score": 0.7},
    )

    path = write_candidate_manifest(manifest, tmp_path / "candidate_manifest.json")
    loaded = load_candidate_manifest(path)

    assert loaded["manifest_type"] == "teacher_candidate_manifest"
    assert loaded["total_candidates"] == 1
    assert loaded["config"]["min_score"] == 0.7


def test_candidate_manifest_rejects_wrong_manifest_type(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"manifest_type": "other"}', encoding="utf-8")

    with pytest.raises(ValueError):
        load_candidate_manifest(path)
