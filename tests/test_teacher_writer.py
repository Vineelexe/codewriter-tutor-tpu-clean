from __future__ import annotations

from cw360.teacher.structured_record import StructuredTrainingRecord, stable_hash, utc_now_iso
from cw360.teacher.writer import StructuredShardWriter, read_structured_shard


def _record(identifier: str = "synth-1") -> StructuredTrainingRecord:
    return StructuredTrainingRecord(
        synthetic_id=identifier,
        source_candidate_id="cand-1",
        source="debugbench",
        record_type="structured",
        task_type="debugging",
        difficulty="beginner",
        skills=["errors"],
        user_prompt="Debug this Python function.",
        assistant_response="The Python function needs an explicit edge-case guard before division.",
        train_stage="instruction_tune",
        loss_weight=1.0,
        teacher_provider="mock",
        teacher_model="mock-teacher",
        generation_time=utc_now_iso(),
        input_hash=stable_hash("in"),
        output_hash=stable_hash("out"),
        metadata={"test": True},
    )


def test_structured_shard_writer_flushes_jsonl(tmp_path) -> None:
    writer = StructuredShardWriter(tmp_path, shard_size=2)
    assert writer.write(_record("synth-1")) == []
    flushed = writer.write(_record("synth-2"))

    assert flushed[0].count == 2
    rows = read_structured_shard(tmp_path / flushed[0].path)
    assert [row.synthetic_id for row in rows] == ["synth-1", "synth-2"]
