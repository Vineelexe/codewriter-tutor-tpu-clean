from __future__ import annotations

import numpy as np

from cw360.prepack.config import load_prepack_config
from cw360.prepack.verify import verify_prepacked_dataset
from scripts.make_tiny_prepacked_dataset import make_tiny_prepacked_dataset


def test_tiny_prepacked_dataset_writes_verifiable_uint16_shards(tmp_path) -> None:
    config = load_prepack_config("configs/prepack_tpu_1024.yaml")
    make_tiny_prepacked_dataset(
        config=config,
        output_dir=tmp_path,
        num_sequences=16,
    )

    report = verify_prepacked_dataset(tmp_path)
    train = np.load(tmp_path / "train" / "chunk_000000.npy")

    assert train.shape == (16, 1025)
    assert train.dtype == np.uint16
    assert report.shard_count == 3
    assert report.total_sequences == 48
