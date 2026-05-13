from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cw360.checkpoint.metadata import CheckpointMetadata, DataSnapshot
from cw360.constants import (
    VALID_MODEL_SIZE_LABELS,
    VALID_PLATFORMS,
    VALID_TRAINERS,
    VALID_TRAINING_STAGES,
)


class CheckpointValidationError(ValueError):
    """Raised when checkpoint metadata or payload is incompatible with a resume target."""


def validate_metadata(metadata: CheckpointMetadata) -> None:
    if metadata.model_size_label not in VALID_MODEL_SIZE_LABELS:
        raise CheckpointValidationError(f"invalid model_size_label: {metadata.model_size_label}")
    if metadata.trained_by not in VALID_TRAINERS:
        raise CheckpointValidationError(f"invalid trained_by: {metadata.trained_by}")
    if metadata.platform not in VALID_PLATFORMS:
        raise CheckpointValidationError(f"invalid platform: {metadata.platform}")
    if metadata.training_stage not in VALID_TRAINING_STAGES:
        raise CheckpointValidationError(f"invalid training_stage: {metadata.training_stage}")
    if metadata.backend not in {"cpu", "cuda", "xla_tpu"}:
        raise CheckpointValidationError(f"invalid backend: {metadata.backend}")
    if metadata.step < 0:
        raise CheckpointValidationError("step must be non-negative")
    if metadata.tokens_seen < 0 or metadata.sequences_seen < 0:
        raise CheckpointValidationError("tokens_seen and sequences_seen must be non-negative")
    validate_data_snapshot(metadata.data_snapshot)
    if metadata.backend == "xla_tpu" and metadata.tpu_data_cursor is None:
        raise CheckpointValidationError("xla_tpu checkpoints require tpu_data_cursor metadata")
    if metadata.tpu_data_cursor is not None:
        cursor = metadata.tpu_data_cursor
        if cursor.global_step != metadata.step:
            raise CheckpointValidationError("tpu_data_cursor.global_step must match checkpoint step")
        if cursor.tokens_seen != metadata.tokens_seen:
            raise CheckpointValidationError("tpu_data_cursor.tokens_seen must match metadata")
        if cursor.sequences_seen != metadata.sequences_seen:
            raise CheckpointValidationError("tpu_data_cursor.sequences_seen must match metadata")
        if not cursor.drop_last:
            raise CheckpointValidationError("real TPU prepacked training requires drop_last=True")


def validate_data_snapshot(snapshot: DataSnapshot) -> None:
    required_strings = {
        "datasets_config_hash": snapshot.datasets_config_hash,
        "mixture_config_hash": snapshot.mixture_config_hash,
        "tokenizer_id": snapshot.tokenizer_id,
        "prepacked_dataset_name": snapshot.prepacked_dataset_name,
        "prepacked_dataset_version": snapshot.prepacked_dataset_version,
        "prepacked_manifest_hash": snapshot.prepacked_manifest_hash,
        "shard_manifest_hash": snapshot.shard_manifest_hash,
        "token_dtype": snapshot.token_dtype,
    }
    missing = [name for name, value in required_strings.items() if not value]
    if missing:
        raise CheckpointValidationError(f"data_snapshot missing fields: {', '.join(missing)}")
    if snapshot.seq_len_plus_one <= 1:
        raise CheckpointValidationError("data_snapshot.seq_len_plus_one must be greater than 1")
    if not snapshot.shard_file_list:
        raise CheckpointValidationError("data_snapshot.shard_file_list must not be empty")


def validate_model_compatibility(model: Any, metadata: CheckpointMetadata, config: Mapping[str, Any]) -> None:
    model_config = getattr(model, "config", None)
    if model_config is None:
        return
    model_dump = model_config.model_dump() if hasattr(model_config, "model_dump") else vars(model_config)
    if str(model_dump.get("size_label")) != metadata.model_size_label:
        raise CheckpointValidationError(
            f"model size mismatch: checkpoint={metadata.model_size_label}, "
            f"model={model_dump.get('size_label')}"
        )
    architecture_keys = [
        "vocab_size",
        "max_position_embeddings",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "tie_word_embeddings",
    ]
    for key in architecture_keys:
        if key in config and key in model_dump and config[key] != model_dump[key]:
            raise CheckpointValidationError(
                f"architecture mismatch for {key}: checkpoint={config[key]}, model={model_dump[key]}"
            )


def validate_resume_request(
    metadata: CheckpointMetadata,
    *,
    expected_model_size_label: str | None = None,
    expected_training_stage: str | None = None,
    expected_data_snapshot: DataSnapshot | None = None,
    allow_training_stage_mismatch: bool = False,
    allow_prepacked_dataset_mismatch: bool = False,
) -> None:
    if expected_model_size_label is not None and expected_model_size_label != metadata.model_size_label:
        raise CheckpointValidationError(
            f"model_size_label mismatch: expected={expected_model_size_label}, "
            f"checkpoint={metadata.model_size_label}"
        )
    if (
        expected_training_stage is not None
        and expected_training_stage != metadata.training_stage
        and not allow_training_stage_mismatch
    ):
        raise CheckpointValidationError(
            f"training_stage mismatch: expected={expected_training_stage}, "
            f"checkpoint={metadata.training_stage}"
        )
    if expected_data_snapshot is not None and not allow_prepacked_dataset_mismatch:
        left = prepacked_identity(expected_data_snapshot)
        right = prepacked_identity(metadata.data_snapshot)
        if left != right:
            raise CheckpointValidationError(
                f"prepacked dataset mismatch: expected={left}, checkpoint={right}"
            )


def prepacked_identity(snapshot: DataSnapshot) -> tuple[str, str, str, str, int, str]:
    return (
        snapshot.prepacked_dataset_name,
        snapshot.prepacked_dataset_version,
        snapshot.prepacked_manifest_hash,
        snapshot.shard_manifest_hash,
        snapshot.seq_len_plus_one,
        snapshot.token_dtype,
    )
