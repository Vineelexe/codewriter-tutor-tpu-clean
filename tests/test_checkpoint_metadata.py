from __future__ import annotations

import json

from cw360.checkpoint import (
    CheckpointMetadata,
    DataSnapshot,
    TPUDataCursor,
    metadata_path_for_checkpoint,
    read_metadata_json,
    write_metadata_json,
)
from cw360.config import ModelConfig


def tiny_lm_config(**overrides: object) -> ModelConfig:
    data = {
        "config_type": "model",
        "size_label": "tiny",
        "vocab_size": 32,
        "max_position_embeddings": 32,
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 32,
    }
    data.update(overrides)
    return ModelConfig.model_validate(data)


def sample_data_snapshot(**overrides: object) -> DataSnapshot:
    data = {
        "datasets_config_hash": "datasets-hash",
        "mixture_config_hash": "mixture-hash",
        "synthetic_repo_id": "vineel/cw360-synthetic",
        "synthetic_shard_list": ["structured-000.jsonl"],
        "teacher_manifest_id": "teacher-manifest",
        "candidate_manifest_id": "candidate-manifest",
        "split_manifest_id": "split-manifest",
        "tokenizer_id": "bigcode/starcoder2-15b",
        "tokenizer_config_hash": "tokenizer-hash",
        "prepacked_dataset_name": "cw360-prepacked",
        "prepacked_dataset_version": "2026-05-23",
        "prepacked_manifest_hash": "prepacked-hash",
        "shard_manifest_hash": "shard-manifest-hash",
        "shard_file_list": ["train-000.npy", "val-000.npy"],
        "shard_hashes": {"train-000.npy": "aaa", "val-000.npy": "bbb"},
        "seq_len_plus_one": 33,
        "token_dtype": "uint16",
    }
    data.update(overrides)
    return DataSnapshot(**data)


def sample_cursor(**overrides: object) -> TPUDataCursor:
    data = {
        "epoch": 1,
        "global_step": 5,
        "tokens_seen": 320,
        "sequences_seen": 10,
        "shard_order_seed": 123,
        "shard_index": 2,
        "sequence_offset": 7,
        "consumed_sequences_in_current_shard": 7,
        "drop_last": True,
        "batch_size_per_device": 2,
        "num_devices": 8,
        "resume_policy": "resume_exact_cursor",
    }
    data.update(overrides)
    return TPUDataCursor(**data)


def sample_metadata(**overrides: object) -> CheckpointMetadata:
    data = {
        "model_name": "cw360",
        "model_size_label": "tiny",
        "run_id": "run-001",
        "step": 5,
        "train_loss": 4.231,
        "val_loss": 4.5,
        "tokens_seen": 320,
        "sequences_seen": 10,
        "trained_by": "vineel",
        "date": "2026-05-23",
        "platform": "local-cpu-test",
        "training_stage": "base_pretrain",
        "previous_checkpoint": None,
        "dataset_mixture": {"python_code": 0.8, "debug_tutor": 0.2},
        "context_length": 32,
        "optimizer": "adamw",
        "learning_rate": 3.0e-4,
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "precision": "float32",
        "backend": "cpu",
        "git_commit": None,
        "notes": "unit test checkpoint",
        "data_snapshot": sample_data_snapshot(),
        "tpu_data_cursor": None,
    }
    data.update(overrides)
    return CheckpointMetadata(**data)


def test_checkpoint_metadata_json_round_trip(checkpoint_tmp_path) -> None:
    checkpoint_path = checkpoint_tmp_path / "checkpoint.pt"
    metadata = sample_metadata()

    metadata_path = write_metadata_json(metadata, checkpoint_path)
    loaded = read_metadata_json(metadata_path)

    assert metadata_path == metadata_path_for_checkpoint(checkpoint_path)
    assert loaded == metadata
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert raw["data_snapshot"]["prepacked_manifest_hash"] == "prepacked-hash"
    assert raw["tpu_data_cursor"] is None


def test_checkpoint_metadata_supports_tpu_cursor() -> None:
    metadata = sample_metadata(
        backend="xla_tpu",
        platform="kaggle-tpu-background",
        tpu_data_cursor=sample_cursor(),
    )

    loaded = CheckpointMetadata.from_dict(metadata.to_dict())

    assert loaded.tpu_data_cursor is not None
    assert loaded.tpu_data_cursor.shard_order_seed == 123
    assert loaded.data_snapshot.seq_len_plus_one == 33
