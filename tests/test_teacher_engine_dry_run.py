from __future__ import annotations

import json

from cw360.teacher.candidate import TeacherCandidate
from cw360.teacher.engine import TeacherEngineConfig, TeacherGenerationEngine
from cw360.teacher.providers.mock import MockTeacherClient
from cw360.teacher.writer import read_structured_shard


def _write_candidate_shard(path) -> None:
    candidate = TeacherCandidate(
        candidate_id="cand-1",
        source="debugbench",
        task_type="debugging",
        score=0.91,
        input_hash="abc",
        estimated_tokens=20,
        reason_selected="debug_signal",
        quality_tier="A",
        proposed_train_stage="instruction_tune",
        proposed_loss_weight=1.0,
        raw_text="def average(values):\n    return sum(values) / len(values)\n",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(candidate.to_dict()) + "\n", encoding="utf-8")


def _config(tmp_path, *, dry_run: bool) -> TeacherEngineConfig:
    shard = tmp_path / "candidates" / "debugging" / "debugbench_candidates.jsonl"
    _write_candidate_shard(shard)
    return TeacherEngineConfig(
        run_id="run-1",
        provider="mock",
        model="mock-teacher",
        max_requests_per_run=10,
        dry_run=dry_run,
        storage={"local_temp_dir": str(tmp_path / "teacher_out"), "shard_size": 10},
        inputs={"candidate_shards": [str(tmp_path / "candidates" / "candidates.jsonl")]},
    )


def test_engine_dry_run_does_not_call_provider(tmp_path) -> None:
    client = MockTeacherClient()
    engine = TeacherGenerationEngine(_config(tmp_path, dry_run=True), client=client)

    result = engine.run()

    assert result.dry_run is True
    assert result.planned_requests == 1
    assert client.calls == 0


def test_engine_mock_generation_consumes_candidates_and_writes_jsonl(tmp_path) -> None:
    client = MockTeacherClient()
    engine = TeacherGenerationEngine(_config(tmp_path, dry_run=False), client=client)

    result = engine.run()

    assert result.completed == 1
    assert result.failed == 0
    assert client.calls == 1
    shard = next((tmp_path / "teacher_out" / "run-1").glob("structured_*.jsonl"))
    records = read_structured_shard(shard)
    assert records[0].source_candidate_id == "cand-1"
