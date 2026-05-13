from __future__ import annotations

from pathlib import Path

from cw360.data.synthetic import iter_synthetic_jsonl


def test_synthetic_loader_preserves_metadata(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.jsonl"
    path.write_text(
        '{"text": "Explain Python list comprehensions.", '
        '"metadata": {"teacher": "offline"}, "source": "synthetic"}\n',
        encoding="utf-8",
    )
    records = list(iter_synthetic_jsonl(path))
    assert len(records) == 1
    assert records[0].example is not None
    assert records[0].example.metadata["teacher"] == "offline"


def test_synthetic_loader_reports_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.jsonl"
    path.write_text('{"text": "ok"}\nnot-json\n{"metadata": {}}\n', encoding="utf-8")
    records = list(iter_synthetic_jsonl(path))
    assert records[0].ok
    assert records[1].error is not None
    assert "malformed JSON" in records[1].error
    assert records[2].error == "line 3: missing text"
