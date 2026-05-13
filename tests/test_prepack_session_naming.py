from __future__ import annotations

import pytest

from cw360.prepack.session_runner import (
    ensure_session_dir_available,
    next_session_id,
    target_is_complete,
)


def test_next_session_id_starts_at_one(tmp_path) -> None:
    assert next_session_id(tmp_path) == "session_0001"


def test_next_session_id_increments_existing_sessions(tmp_path) -> None:
    (tmp_path / "session_0001").mkdir()

    assert next_session_id(tmp_path) == "session_0002"


def test_explicit_non_empty_session_dir_is_rejected(tmp_path) -> None:
    session_dir = tmp_path / "session_0007"
    session_dir.mkdir()
    (session_dir / "sentinel.txt").write_text("existing output", encoding="utf-8")

    with pytest.raises(FileExistsError, match="non-empty"):
        ensure_session_dir_available(tmp_path, "session_0007")


def test_target_complete_check_uses_existing_valid_sequences() -> None:
    assert target_is_complete(existing_valid_sequences=100, target_total_sequences=100)
    assert target_is_complete(existing_valid_sequences=120, target_total_sequences=100)
    assert not target_is_complete(existing_valid_sequences=99, target_total_sequences=100)
