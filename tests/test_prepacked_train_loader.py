from __future__ import annotations

import numpy as np

from cw360.train.shard_dataloader import PrepackedShardDataLoader
from tests.test_prepack_writer import make_sequence, small_prepack_config
from cw360.prepack.writer import PrepackDatasetWriter


def test_prepacked_loader_mmaps_uint16_shards_and_yields_shifted_long_batches(
    tmp_path,
    monkeypatch,
) -> None:
    config = small_prepack_config(tmp_path, max_seq_len=4, shard_num_sequences=4)
    writer = PrepackDatasetWriter(config=config)
    for index in range(4):
        writer.add_sequence(make_sequence(index * 10), split="train")
    writer.close()

    mmap_modes: list[str | None] = []
    real_load = np.load

    def recording_load(*args, **kwargs):
        mmap_modes.append(kwargs.get("mmap_mode"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr("cw360.train.shard_dataloader.np.load", recording_load)

    loader = PrepackedShardDataLoader(tmp_path, split="train", batch_size=2)
    batch = next(iter(loader))

    assert mmap_modes == ["r"]
    assert batch.input_ids.shape == (2, 4)
    assert batch.labels.shape == (2, 4)
    assert batch.input_ids.dtype.is_floating_point is False
    assert str(batch.input_ids.dtype) == "torch.int64"
    assert torch_shift_matches(batch.input_ids.numpy(), batch.labels.numpy())
    assert batch.tpu_data_cursor is not None
    assert batch.tpu_data_cursor.sequence_offset == 2


def torch_shift_matches(input_ids: np.ndarray, labels: np.ndarray) -> bool:
    return bool((labels[:, :-1] == input_ids[:, 1:]).all())
