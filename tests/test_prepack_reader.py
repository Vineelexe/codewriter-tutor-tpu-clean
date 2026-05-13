from __future__ import annotations

from cw360.prepack.reader import PrepackedShardReader
from cw360.prepack.writer import PrepackDatasetWriter
from tests.test_prepack_writer import make_sequence, small_prepack_config


def test_reader_yields_shifted_input_label_batches_and_cursor(tmp_path) -> None:
    config = small_prepack_config(tmp_path, shard_num_sequences=4)
    writer = PrepackDatasetWriter(config=config)
    for index in range(4):
        writer.add_sequence(make_sequence(index * 10), split="train")
    writer.close()

    reader = PrepackedShardReader(tmp_path, split="train", shard_order_seed=0, mmap=False)
    batch = next(reader.iter_batches(batch_size=2))

    assert batch.input_ids.shape == (2, 4)
    assert batch.labels.shape == (2, 4)
    assert (batch.labels[:, :-1] == batch.input_ids[:, 1:]).all()
    assert batch.cursor.consumed_sequences == 2
    assert batch.cursor.tokens_seen == 2 * config.prepack.max_seq_len


def test_reader_resumes_from_prepacked_cursor_metadata(tmp_path) -> None:
    config = small_prepack_config(tmp_path, shard_num_sequences=4)
    writer = PrepackDatasetWriter(config=config)
    for index in range(4):
        writer.add_sequence(make_sequence(index * 10), split="train")
    writer.close()

    reader = PrepackedShardReader(tmp_path, split="train", shard_order_seed=0, mmap=False)
    first_sequence, cursor = next(reader.iter_sequences())
    resumed_sequence, resumed_cursor = next(reader.iter_sequences(cursor=cursor))

    assert resumed_sequence.tolist() != first_sequence.tolist()
    assert resumed_cursor.consumed_sequences == 2
