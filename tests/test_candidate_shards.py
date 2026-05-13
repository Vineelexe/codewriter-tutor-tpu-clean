from __future__ import annotations

from cw360.teacher.candidate import TeacherCandidate
from cw360.teacher.candidate_shards import CandidateShardWriter, read_candidate_shard


def _candidate(task_type: str = "debugging", source: str = "debugbench") -> TeacherCandidate:
    return TeacherCandidate(
        candidate_id=f"{source}:{task_type}:abc",
        source=source,
        task_type=task_type,
        raw_text="def broken():\n    return 1 / 0\n",
        selected_text=None,
        score=0.91,
        input_hash="abc",
        estimated_tokens=20,
        metadata={"path": "case.py"},
        reason_selected="debug_signal",
        quality_tier="A",
        proposed_train_stage="instruction_tune",
        proposed_loss_weight=1.0,
    )


def test_candidate_shard_writer_groups_by_task_and_source(tmp_path) -> None:
    writer = CandidateShardWriter(tmp_path)
    manifest = writer.write_candidates(
        [
            _candidate("debugging", "debugbench"),
            _candidate("comments", "stack_v2_python"),
        ]
    )

    assert manifest.total_candidates == 2
    assert (tmp_path / "candidate_manifest.json").exists()
    debug_shard = tmp_path / "debugging" / "debugbench_candidates.jsonl"
    comments_shard = tmp_path / "comments" / "stack_v2_python_candidates.jsonl"
    assert debug_shard.exists()
    assert comments_shard.exists()
    assert read_candidate_shard(debug_shard)[0].task_type == "debugging"


def test_candidate_shard_writer_supports_gzip(tmp_path) -> None:
    writer = CandidateShardWriter(tmp_path, gzip_output=True)
    manifest = writer.write_candidates([_candidate()])

    shard_path = tmp_path / manifest.shards[0]["path"]
    assert shard_path.suffix == ".gz"
    assert read_candidate_shard(shard_path)[0].candidate_id.endswith("abc")
