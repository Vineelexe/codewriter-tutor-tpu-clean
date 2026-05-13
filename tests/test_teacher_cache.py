from __future__ import annotations

from cw360.teacher.cache import TeacherResponseCache


def test_teacher_response_cache_round_trips_jsonl(tmp_path) -> None:
    path = tmp_path / "cache.jsonl"
    cache = TeacherResponseCache(path)
    cache.set("hash-1", {"synthetic_id": "synth-1"})

    reloaded = TeacherResponseCache(path)

    assert reloaded.get("hash-1") == {"synthetic_id": "synth-1"}
    assert "hash-1" in reloaded
