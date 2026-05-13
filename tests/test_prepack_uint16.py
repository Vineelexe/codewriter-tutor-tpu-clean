from __future__ import annotations

import pytest

from cw360.prepack.writer import PrepackDatasetWriter
from tests.test_prepack_writer import make_sequence, small_prepack_config


def test_prepack_writer_rejects_token_id_overflow(tmp_path) -> None:
    config = small_prepack_config(tmp_path)
    writer = PrepackDatasetWriter(config=config)
    sequence = make_sequence(1)
    bad = type(sequence)(
        token_ids=[1, 2, 3, 4, 65_536],
        split_key=sequence.split_key,
        source_counts=sequence.source_counts,
        task_type_counts=sequence.task_type_counts,
        training_stage_counts=sequence.training_stage_counts,
        quality_tier_counts=sequence.quality_tier_counts,
        packing_mode_counts=sequence.packing_mode_counts,
        source_example_counts=sequence.source_example_counts,
    )

    with pytest.raises(ValueError, match="uint16"):
        writer.add_sequence(bad, split="train")


def test_prepack_writer_accepts_highest_uint16_token_id(tmp_path) -> None:
    config = small_prepack_config(tmp_path)
    writer = PrepackDatasetWriter(config=config)
    sequence = make_sequence(1)
    ok = type(sequence)(
        token_ids=[1, 2, 3, 4, 65_535],
        split_key=sequence.split_key,
        source_counts=sequence.source_counts,
        task_type_counts=sequence.task_type_counts,
        training_stage_counts=sequence.training_stage_counts,
        quality_tier_counts=sequence.quality_tier_counts,
        packing_mode_counts=sequence.packing_mode_counts,
        source_example_counts=sequence.source_example_counts,
    )

    writer.add_sequence(ok, split="train")
    manifest = writer.close()

    assert manifest["total_sequences"] == 1
