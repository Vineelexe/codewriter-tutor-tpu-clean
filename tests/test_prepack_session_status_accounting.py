from __future__ import annotations

from cw360.prepack.manifest import read_json, write_json
from cw360.prepack.session_runner import scan_prepack_sessions, target_is_complete
from cw360.prepack.writer import PrepackDatasetWriter
from scripts.finalize_prepack_sessions import finalize_sessions
from tests.test_prepack_writer import make_sequence, small_prepack_config


def test_only_completed_sessions_count_toward_valid_sequences(tmp_path) -> None:
    sessions_root = tmp_path / "sessions"
    _write_session(sessions_root / "session_0001", status="completed", sequences=2)
    _write_session(sessions_root / "session_0002", status="running", sequences=3)
    _write_session(sessions_root / "session_0003", status="failed", sequences=4)
    _write_session(sessions_root / "session_0004", status=None, sequences=5)

    scan = scan_prepack_sessions(sessions_root)

    assert [session.session_id for session in scan.valid_sessions] == ["session_0001"]
    assert scan.total_valid_sequences == 2
    assert len(scan.ignored_sessions) == 3
    assert not target_is_complete(scan.total_valid_sequences, target_total_sequences=3)


def test_target_is_not_complete_when_only_incomplete_sessions_have_enough_sequences(
    tmp_path,
) -> None:
    sessions_root = tmp_path / "sessions"
    _write_session(sessions_root / "session_0001", status="running", sequences=4)
    _write_session(sessions_root / "session_0002", status="failed", sequences=4)

    scan = scan_prepack_sessions(sessions_root)

    assert scan.total_valid_sequences == 0
    assert not target_is_complete(scan.total_valid_sequences, target_total_sequences=8)


def test_finalizer_uses_only_completed_valid_sessions(tmp_path) -> None:
    sessions_root = tmp_path / "sessions"
    _write_session(sessions_root / "session_0001", status="completed", sequences=2)
    _write_session(sessions_root / "session_0002", status="running", sequences=4)
    _write_session(sessions_root / "session_0003", status="failed", sequences=4)
    scan = scan_prepack_sessions(sessions_root)

    manifest = finalize_sessions(scan.valid_sessions, output_dir=tmp_path / "final")

    assert [session.session_id for session in scan.valid_sessions] == ["session_0001"]
    assert manifest["total_sequences"] == 2
    assert read_json(tmp_path / "final" / "manifest.json")["total_sequences"] == 2


def _write_session(session_dir, *, status: str | None, sequences: int) -> None:
    config = small_prepack_config(session_dir, shard_num_sequences=max(1, sequences))
    writer = PrepackDatasetWriter(config=config)
    for index in range(sequences):
        writer.add_sequence(make_sequence(index), split="train")
    writer.close()
    if status is not None:
        write_json(
            session_dir / "session_info.json",
            {
                "session_id": session_dir.name,
                "status": status,
                "final_session_sequences": sequences,
            },
        )
