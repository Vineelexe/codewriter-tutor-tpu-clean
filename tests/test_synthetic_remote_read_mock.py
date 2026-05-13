from __future__ import annotations

from cw360.synthetic.remote_read import iter_hf_dataset_rows, pull_synthetic_rows_to_dir
from tests.test_synthetic_schema import sample_payload


def test_remote_read_accepts_mock_dataset_loader() -> None:
    def loader(repo_id: str, **kwargs):
        assert repo_id == "private/synthetic"
        assert kwargs["split"] == "train"
        assert kwargs["streaming"] is True
        return [sample_payload()]

    rows = list(iter_hf_dataset_rows("private/synthetic", dataset_loader=loader))

    assert rows[0]["synthetic_id"] == "synth-1"


def test_pull_synthetic_rows_writes_local_shards(tmp_path) -> None:
    written = pull_synthetic_rows_to_dir(
        [sample_payload(synthetic_id="synth-1"), sample_payload(synthetic_id="synth-2")],
        tmp_path,
        shard_size=1,
    )

    assert [path.name for path in written] == ["structured_00000.jsonl", "structured_00001.jsonl"]
    assert written[0].read_text(encoding="utf-8").startswith("{")
