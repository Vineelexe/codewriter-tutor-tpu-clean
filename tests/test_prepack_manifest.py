from __future__ import annotations

from cw360.prepack.manifest import read_json
from cw360.prepack.writer import LOSS_WEIGHT_HANDLING_POLICY, PrepackDatasetWriter
from tests.test_prepack_writer import make_sequence, small_prepack_config


def test_global_manifest_records_hashes_splits_and_loss_policy(tmp_path) -> None:
    config = small_prepack_config(tmp_path, shard_num_sequences=2)
    writer = PrepackDatasetWriter(config=config, data_snapshot_hash="snapshot-hash")
    writer.add_sequence(make_sequence(1), split="train")
    writer.add_sequence(make_sequence(2, source="structured_synthetic_instructions"), split="train")
    manifest = writer.close()
    on_disk = read_json(tmp_path / "manifest.json")

    assert on_disk["manifest_hash"] == manifest["manifest_hash"]
    assert on_disk["data_snapshot_hash"] == "snapshot-hash"
    assert on_disk["loss_weight_handling_policy"] == LOSS_WEIGHT_HANDLING_POLICY
    assert on_disk["shard_hashes"]["train"]["train/chunk_000000.npy"]
    assert on_disk["total_tokens_for_training"] == 2 * config.prepack.max_seq_len
    assert on_disk["validation"]["fail_on_token_overflow"] is True
