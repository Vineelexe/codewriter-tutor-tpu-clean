from __future__ import annotations

from cw360.teacher.remote_store import MockRemoteStore


def test_mock_remote_store_records_uploaded_files(tmp_path) -> None:
    (tmp_path / "structured_00000.jsonl").write_text("{}", encoding="utf-8")
    store = MockRemoteStore()

    result = store.upload_dir(tmp_path)

    assert result.uploaded is True
    assert result.destination == "mock://remote-store"
    assert result.files == ["structured_00000.jsonl"]
