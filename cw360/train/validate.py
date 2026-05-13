from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cw360.checkpoint.metadata import DataSnapshot
from cw360.checkpoint.validate import prepacked_identity
from cw360.config import ModelConfig, load_config, load_model_config
from cw360.constants import (
    VALID_MODEL_SIZE_LABELS,
    VALID_PLATFORMS,
    VALID_TRAINERS,
    VALID_TRAINING_STAGES,
)
from cw360.model.count import build_parameter_report, validate_size_label_count
from cw360.prepack.manifest import read_json
from cw360.tokenizer.special_tokens import UINT16_TOKEN_ID_LIMIT, is_starcoder2_tokenizer_id
from cw360.train.stages import ensure_trainable_stage, validate_training_stage

TPU_PLATFORM_LABEL = "kaggle-tpu-background"
REQUIRED_TPU_CURSOR_FIELDS = frozenset(
    {
        "epoch",
        "shard_order_seed",
        "shard_index",
        "sequence_offset",
        "consumed_sequences",
        "tokens_seen",
    }
)


def validate_stage_for_training(stage: str) -> None:
    ensure_trainable_stage(validate_training_stage(stage))


def validate_tiny_cpu_model(model_config: ModelConfig) -> None:
    if model_config.size_label != "tiny":
        raise ValueError("local CPU training must use a tiny model config")


def validate_model_size_label(size_label: str) -> None:
    if size_label not in VALID_MODEL_SIZE_LABELS:
        raise ValueError(f"size_label must be one of {sorted(VALID_MODEL_SIZE_LABELS)}")


def validate_static_shape_rules(
    *,
    max_seq_len: int,
    batch_size: int,
    drop_last: bool,
    static_shapes: bool,
) -> None:
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not static_shapes:
        raise ValueError("TPU/prepacked training requires static_shapes=True")
    if not drop_last:
        raise ValueError("TPU/prepacked training requires drop_last=True")


def validate_manifest_matches_snapshot(
    *,
    checkpoint_snapshot: DataSnapshot,
    current_snapshot: DataSnapshot,
    allow_override: bool = False,
) -> None:
    if allow_override:
        return
    if prepacked_identity(checkpoint_snapshot) != prepacked_identity(current_snapshot):
        raise ValueError("checkpoint data_snapshot is not compatible with prepacked manifest")


def require_manifest_field(manifest: Mapping[str, Any], key: str) -> Any:
    if key not in manifest:
        raise ValueError(f"prepacked manifest missing {key!r}")
    return manifest[key]


def validate_model_architecture(
    model_config: ModelConfig,
    *,
    require_explicit_mlp_hidden_size: bool = False,
    parameter_tolerance: float = 0.01,
) -> None:
    if model_config.num_attention_heads % model_config.num_key_value_heads != 0:
        raise ValueError("n_query_heads must be divisible by n_kv_heads")
    if model_config.hidden_size != model_config.num_attention_heads * model_config.head_dim:
        raise ValueError("hidden_size must equal n_query_heads * head_dim")
    if not model_config.tie_word_embeddings:
        raise ValueError("tied_embeddings must be true")
    if not model_config.qkv_bias:
        raise ValueError("qkv_bias must be true")
    if require_explicit_mlp_hidden_size and model_config.mlp_hidden_size is None:
        raise ValueError("mlp_hidden_size must be configured explicitly")
    if (
        model_config.mlp_hidden_size is not None
        and model_config.mlp_hidden_size != model_config.intermediate_size
    ):
        raise ValueError("mlp_hidden_size must match intermediate_size")
    if model_config.vocab_size >= UINT16_TOKEN_ID_LIMIT:
        raise ValueError("tokenizer vocab must fit uint16 token IDs")
    if not is_starcoder2_tokenizer_id(model_config.tokenizer_name):
        raise ValueError("tokenizer_id must reference a StarCoder2 tokenizer")

    report = build_parameter_report(model_config)
    if not validate_size_label_count(report, tolerance=parameter_tolerance):
        raise ValueError(
            "actual parameter count is not close to expected candidate range "
            f"for {model_config.size_label}: {report.total_parameters}"
        )


def validate_mixture_sums_to_one(weights: Mapping[str, Any], *, label: str) -> None:
    if not weights:
        raise ValueError(f"{label} must not be empty")
    numeric = {str(name): float(weight) for name, weight in weights.items()}
    if any(weight < 0.0 for weight in numeric.values()):
        raise ValueError(f"{label} weights must be non-negative")
    total = sum(numeric.values())
    if abs(total - 1.0) > 1.0e-9:
        raise ValueError(f"{label} weights must sum to 1.0")


def validate_teacher_candidates_config(config: Mapping[str, Any]) -> None:
    task_mix = _required_mapping(config, "task_mix")
    if "refactor" not in task_mix or float(task_mix["refactor"]) <= 0.0:
        raise ValueError("teacher_candidates task_mix must include refactor")
    validate_mixture_sums_to_one(task_mix, label="teacher_candidates task_mix")
    if not bool(config.get("offline_only", False)):
        raise ValueError("teacher candidate generation config must be offline_only")


def validate_train_config_data_snapshot(config: Mapping[str, Any]) -> None:
    snapshot = _required_mapping(config, "data_snapshot")
    if not bool(snapshot.get("required", False)):
        raise ValueError("data_snapshot.required must be true for train configs")
    source = str(snapshot.get("source", ""))
    if not source:
        raise ValueError("data_snapshot.source is required")


def validate_tpu_train_config(config_path: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config_path, Mapping):
        config = dict(config_path)
        root = Path.cwd()
    else:
        path = Path(config_path)
        config = load_config(path)
        root = path.resolve().parents[1] if path.parent.name == "configs" else path.resolve().parent

    if config.get("config_type") != "train":
        raise ValueError("TPU audit requires a train config")
    if config.get("platform") not in VALID_PLATFORMS:
        raise ValueError("platform label is invalid")
    if str(config.get("platform")) != TPU_PLATFORM_LABEL:
        raise ValueError(f"TPU train configs must use platform {TPU_PLATFORM_LABEL!r}")
    if config.get("training_stage") not in VALID_TRAINING_STAGES:
        raise ValueError("training_stage is invalid")
    validate_stage_for_training(str(config["training_stage"]))
    validate_train_config_data_snapshot(config)

    model_path = _resolve_repo_path(root, str(_required(config, "model_config_path")))
    model_config = load_model_config(model_path)
    validate_model_architecture(model_config, require_explicit_mlp_hidden_size=True)

    tokenizer_id = str(_required(config, "tokenizer_id"))
    if tokenizer_id != model_config.tokenizer_name:
        raise ValueError("train tokenizer_id must match model tokenizer_name")
    if not is_starcoder2_tokenizer_id(tokenizer_id):
        raise ValueError("tokenizer_id must reference a StarCoder2 tokenizer")

    max_seq_len = int(_required(config, "max_seq_len"))
    if max_seq_len > model_config.max_position_embeddings:
        raise ValueError("context length is larger than model max_position_embeddings")
    if max_seq_len not in {1024, 2048}:
        raise ValueError("TPU train configs must use max_seq_len 1024 or 2048")

    _validate_prepacked_paths(config)
    _validate_batch_and_shape_fields(config, max_seq_len=max_seq_len)
    _validate_optimizer_and_schedule(config)
    _validate_tpu_precision(config)
    _validate_runtime_guards(config)
    _validate_checkpointing(config)
    _validate_durable_storage(config)
    _validate_tpu_data_cursor(config)
    _validate_optional_manifest(config)
    return config


def _validate_prepacked_paths(config: Mapping[str, Any]) -> None:
    if not any(
        key in config and str(config[key])
        for key in ("prepacked_shards_path", "prepacked_dataset_path", "kaggle_input_path")
    ):
        raise ValueError("TPU train config requires a prepacked dataset or Kaggle input path")
    _required(config, "prepacked_manifest_path")
    if "shard_manifest_hash_expected" not in config:
        raise ValueError("shard_manifest_hash_expected must be configured, even if unknown")


def _validate_batch_and_shape_fields(config: Mapping[str, Any], *, max_seq_len: int) -> None:
    fixed_batch_size = int(_required(config, "fixed_batch_size"))
    micro_batch_size = int(_required(config, "micro_batch_size"))
    if fixed_batch_size != micro_batch_size:
        raise ValueError("fixed_batch_size must match micro_batch_size")
    validate_static_shape_rules(
        max_seq_len=max_seq_len,
        batch_size=fixed_batch_size,
        drop_last=bool(config.get("drop_last", False)),
        static_shapes=bool(config.get("static_shapes", False)),
    )
    if bool(config.get("dynamic_padding", True)):
        raise ValueError("TPU train configs must disable dynamic padding")


def _validate_optimizer_and_schedule(config: Mapping[str, Any]) -> None:
    if str(config.get("optimizer", "")).lower() != "adamw":
        raise ValueError("TPU train configs must use AdamW optimizer")
    if str(config.get("schedule_type", "")).lower() != "wsd":
        raise ValueError("TPU train configs must use warmup/WSD schedule")
    if float(_required(config, "learning_rate")) <= 0.0:
        raise ValueError("learning_rate must be positive")
    if int(_required(config, "warmup_steps")) < 0:
        raise ValueError("warmup_steps must be non-negative")
    if float(_required(config, "max_grad_norm")) <= 0.0:
        raise ValueError("gradient clipping must be configured with max_grad_norm > 0")
    if "gradient_checkpointing" not in config:
        raise ValueError("gradient_checkpointing must be configurable")
    if bool(config.get("torch_compile", True)):
        raise ValueError("torch_compile must be disabled for TPU configs unless explicitly tested")


def _validate_tpu_precision(config: Mapping[str, Any]) -> None:
    precision = str(_required(config, "precision")).lower()
    if precision != "bf16":
        raise ValueError("TPU precision must be bf16/XLA-compatible, not CUDA fp16 logic")
    for key in ("use_fp16_grad_scaler", "fp16_grad_scaler", "cuda_amp", "use_cuda_amp"):
        if bool(config.get(key, False)):
            raise ValueError("CUDA-only fp16 GradScaler logic is not allowed in TPU configs")
    if str(config.get("backend", "xla_tpu")).lower() == "cuda":
        raise ValueError("TPU configs must not select a CUDA backend")


def _validate_runtime_guards(config: Mapping[str, Any]) -> None:
    for key in (
        "dataloader_num_workers",
        "prefetch_factor",
        "save_every_n_minutes",
        "session_time_limit_minutes",
        "save_before_exit_minutes",
    ):
        if int(_required(config, key)) <= 0:
            raise ValueError(f"{key} must be positive")


def _validate_checkpointing(config: Mapping[str, Any]) -> None:
    if str(_required(config, "checkpoint_dir")) == "":
        raise ValueError("checkpoint_dir must be configurable")
    if int(_required(config, "checkpoint_interval_steps")) <= 0:
        raise ValueError("checkpoint_interval_steps must be positive")
    if int(_required(config, "eval_interval_steps")) <= 0:
        raise ValueError("eval_interval_steps must be positive")
    trainer = str(config.get("trainer_name") or config.get("trained_by") or "")
    if trainer not in VALID_TRAINERS:
        raise ValueError(f"trainer name must be one of {sorted(VALID_TRAINERS)}")


def _validate_durable_storage(config: Mapping[str, Any]) -> None:
    durable = _required_mapping(config, "durable_checkpoint_store")
    if not str(durable.get("type", "")):
        raise ValueError("durable_checkpoint_store.type is required")
    if "path_prefix" not in durable:
        raise ValueError("durable_checkpoint_store.path_prefix is required")
    if "private" not in durable:
        raise ValueError("durable_checkpoint_store.private is required")


def _validate_tpu_data_cursor(config: Mapping[str, Any]) -> None:
    cursor = _required_mapping(config, "tpu_data_cursor")
    if not bool(cursor.get("enabled", False)):
        raise ValueError("tpu_data_cursor.enabled must be true")
    fields = {str(field) for field in cursor.get("required_fields", [])}
    missing = REQUIRED_TPU_CURSOR_FIELDS.difference(fields)
    if missing:
        raise ValueError(f"tpu_data_cursor missing required fields: {sorted(missing)}")


def _validate_optional_manifest(config: Mapping[str, Any]) -> None:
    manifest_path = Path(str(config["prepacked_manifest_path"]))
    if not manifest_path.exists():
        return
    manifest = read_json(manifest_path)
    if int(manifest["seq_len_plus_one"]) != int(manifest["max_seq_len"]) + 1:
        raise ValueError("seq_len_plus_one must equal max_seq_len + 1")
    if str(manifest["dtype"]) != "uint16":
        raise ValueError("shard dtype must be uint16")
    expected = config.get("shard_manifest_hash_expected")
    if expected not in (None, "") and str(manifest.get("shard_manifest_hash")) != str(expected):
        raise ValueError("prepacked shard manifest hash does not match expected value")


def _resolve_repo_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _required(config: Mapping[str, Any], key: str) -> Any:
    if key not in config:
        raise ValueError(f"missing required config key {key!r}")
    return config[key]


def _required_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = _required(config, key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value
