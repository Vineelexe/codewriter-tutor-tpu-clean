from __future__ import annotations

import numpy as np

from cw360.prepack.config import (
    PrepackFactoryConfig,
    PrepackSection,
    QualityConfig,
    SplitsConfig,
    ValidationConfig,
)
from cw360.prepack.manifest import read_json
from cw360.prepack.writer import PrepackDatasetWriter, PrepackedSequence


def small_prepack_config(tmp_path, *, max_seq_len: int = 4, shard_num_sequences: int = 4):
    return PrepackFactoryConfig(
        prepack=PrepackSection(
            dataset_name="cw-tiny-prepack",
            dataset_version="test",
            model_size_target="selectable",
            tokenizer_id="bigcode/starcoder2-15b",
            max_seq_len=max_seq_len,
            seq_len_plus_one=max_seq_len + 1,
            token_dtype="uint16",
            shard_format="npy",
            shard_num_sequences=shard_num_sequences,
            rolling_shuffle_buffer_sequences=2,
            rolling_shuffle_buffer_memory_guard=False,
            rolling_shuffle_buffer_max_ram_fraction=0.25,
            minimum_shuffle_buffer_sequences=1,
            output_dir=str(tmp_path),
            seed=7,
            drop_incomplete_sequence=True,
            verify_uint16=True,
            write_per_shard_metadata=True,
            write_global_manifest=True,
            write_kaggle_dataset_metadata=True,
            kaggle_owner="vineel",
        ),
        mixture={"stack_v2_python": 0.5, "structured_synthetic_instructions": 0.5},
        splits=SplitsConfig(
            train_fraction=0.8,
            val_fraction=0.1,
            test_fraction=0.1,
            split_by="stable_hash",
            synthetic_split_key="source_candidate_id",
        ),
        quality=QualityConfig(
            tier_a_weight=1.0,
            tier_b_weight=0.5,
            rescue_policy="excerpt_or_skip_with_report",
        ),
        validation=ValidationConfig(
            shard_mixture_tolerance_abs=0.5,
            fail_on_token_overflow=True,
            fail_on_empty_shard=True,
            fail_on_missing_manifest=True,
            fail_if_split_fractions_do_not_sum_to_one=True,
            fail_if_atomic_examples_cross_sequence_boundary=True,
        ),
    )


def make_sequence(index: int, *, source: str = "stack_v2_python", seq_len: int = 5):
    task_type = "python_code" if source == "stack_v2_python" else "instruction"
    packing_mode = "continuous_pack" if task_type == "python_code" else "example_atomic_pack"
    return PrepackedSequence(
        token_ids=[index + offset for offset in range(seq_len)],
        split_key=f"{source}:{index}",
        source_counts={source: 1},
        task_type_counts={task_type: 1},
        training_stage_counts={"base_pretrain": 1},
        quality_tier_counts={"A": 1},
        packing_mode_counts={packing_mode: 1},
        source_example_counts={source: 1},
    )


def test_prepack_writer_writes_npy_shard_and_metadata(tmp_path) -> None:
    config = small_prepack_config(tmp_path, shard_num_sequences=2)
    writer = PrepackDatasetWriter(config=config)

    writer.add_sequence(make_sequence(1), split="train")
    writer.add_sequence(
        make_sequence(10, source="structured_synthetic_instructions"),
        split="train",
    )
    manifest = writer.close()

    shard_path = tmp_path / "train" / "chunk_000000.npy"
    metadata_path = tmp_path / "train" / "chunk_000000.json"
    array = np.load(shard_path)
    metadata = read_json(metadata_path)

    assert array.shape == (2, 5)
    assert array.dtype == np.uint16
    assert metadata["shape"] == [2, 5]
    assert metadata["dtype"] == "uint16"
    assert metadata["split"] == "train"
    assert metadata["source_example_counts"]["stack_v2_python"] == 1
    assert manifest["shard_file_list"]["train"] == ["train/chunk_000000.npy"]
    assert (tmp_path / "dataset-metadata.json").exists()
