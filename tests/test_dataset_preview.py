from __future__ import annotations

from pathlib import Path

from scripts.dataset_preview import preview_dataset


def test_dataset_preview_prints_source_tier_reason_and_status(tmp_path: Path) -> None:
    data_path = tmp_path / "synthetic.jsonl"
    data_path.write_text(
        '{"text": "def add(a, b):\\n    return a + b", '
        '"source": "synthetic", "metadata": {"id": "1"}}\n',
        encoding="utf-8",
    )
    config_path = tmp_path / "datasets.yaml"
    config_path.write_text(
        f"""
config_type: datasets
sources:
  synthetic:
    local_path: {data_path.as_posix()}
    streaming: true
    extractor: synthetic
""",
        encoding="utf-8",
    )

    rows = preview_dataset(config_path, "synthetic", 1)
    assert len(rows) == 1
    assert "source=synthetic" in rows[0]
    assert "tier=" in rows[0]
    assert "reason=" in rows[0]
    assert "estimated_tokens=" in rows[0]
