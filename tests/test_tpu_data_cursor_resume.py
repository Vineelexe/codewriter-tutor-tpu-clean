from __future__ import annotations

from dataclasses import replace

import pytest

from cw360.train.cpu_trainer import CPUTinyTrainer
from cw360.train.shard_dataloader import (
    PrepackedShardDataLoader,
    data_snapshot_from_prepacked_manifest,
)
from cw360.prepack.writer import PrepackDatasetWriter
from tests.test_prepack_writer import make_sequence, small_prepack_config
from tests.test_training_step import tiny_train_config


def test_prepacked_loader_resumes_exactly_from_tpu_data_cursor(tmp_path) -> None:
    write_prepacked_fixture(tmp_path, num_sequences=6)
    first_loader = PrepackedShardDataLoader(tmp_path, split="train", batch_size=2)
    iterator = iter(first_loader)
    first = next(iterator)
    second = next(iterator)

    resumed_loader = PrepackedShardDataLoader(
        tmp_path,
        split="train",
        batch_size=2,
        cursor=first.tpu_data_cursor,
    )
    resumed = next(iter(resumed_loader))

    assert resumed.input_ids.equal(second.input_ids)
    assert resumed.labels.equal(second.labels)
    assert resumed.tpu_data_cursor is not None
    assert resumed.tpu_data_cursor.sequence_offset == 4


def test_prepacked_checkpoint_resume_validates_data_snapshot(tmp_path) -> None:
    prepacked_dir = tmp_path / "prepacked"
    write_prepacked_fixture(prepacked_dir, num_sequences=4)
    snapshot = data_snapshot_from_prepacked_manifest(prepacked_dir)
    config = tiny_train_config(tmp_path, max_steps=1)

    def factory(cursor):
        return iter(PrepackedShardDataLoader(prepacked_dir, split="train", batch_size=2, cursor=cursor))

    first = CPUTinyTrainer(
        config=config,
        output_dir=tmp_path / "ckpts",
        batch_factory=factory,
        data_snapshot=snapshot,
    )
    result = first.train()
    incompatible = replace(snapshot, prepacked_dataset_version="different")

    with pytest.raises(ValueError, match="prepacked dataset mismatch"):
        CPUTinyTrainer(
            config=tiny_train_config(tmp_path, max_steps=2),
            output_dir=tmp_path / "ckpts",
            batch_factory=factory,
            data_snapshot=incompatible,
            resume_checkpoint=result.checkpoints[-1],
        )


def write_prepacked_fixture(tmp_path, *, num_sequences: int) -> None:
    config = small_prepack_config(tmp_path, max_seq_len=8, shard_num_sequences=num_sequences)
    writer = PrepackDatasetWriter(config=config)
    for index in range(num_sequences):
        writer.add_sequence(make_sequence(index * 10, seq_len=9), split="train")
        writer.add_sequence(make_sequence(index * 20, seq_len=9), split="val")
    writer.close()
