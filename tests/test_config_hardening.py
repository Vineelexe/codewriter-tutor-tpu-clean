from __future__ import annotations

from pathlib import Path

import pytest

from cw360.config import ConfigError, load_config, load_model_config, load_yaml, parse_config
from cw360.train.validate import validate_tpu_train_config, validate_train_config_data_snapshot

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

TPU_1024_CONFIGS = (
    "train_tpu_354m_1024.yaml",
    "train_tpu_420m_1024.yaml",
    "train_tpu_480m_1024.yaml",
)


def test_candidate_model_configs_expose_hardened_architecture_fields() -> None:
    for filename in ("model_354m.yaml", "model_420m.yaml", "model_480m.yaml"):
        config = load_model_config(CONFIG_DIR / filename)

        assert config.tokenizer_name == "bigcode/starcoder2-15b"
        assert config.tie_word_embeddings is True
        assert config.qkv_bias is True
        assert config.hidden_size == 768
        assert config.max_position_embeddings == 1024
        assert config.activation == "swiglu"
        assert config.norm == "rmsnorm"
        assert config.position_encoding == "rope"
        assert config.attention == "gqa"
        assert config.mlp_hidden_size == config.intermediate_size
        assert config.hidden_size == config.num_attention_heads * config.head_dim
        assert config.num_attention_heads % config.num_key_value_heads == 0


def test_candidate_model_configs_keep_explicit_deep_thin_yaml_fields() -> None:
    expected_layers = {
        "model_354m.yaml": 46,
        "model_420m.yaml": 56,
        "model_480m.yaml": 64,
    }
    for filename, n_layers in expected_layers.items():
        raw = load_yaml(CONFIG_DIR / filename)

        assert raw["hidden_size"] == 768
        assert raw["n_layers"] == n_layers
        assert raw["n_query_heads"] == 12
        assert raw["n_kv_heads"] == 4
        assert raw["head_dim"] == 64
        assert raw["mlp_hidden_size"] == 2304
        assert raw["tied_embeddings"] is True
        assert raw["qkv_bias"] is True
        assert raw["max_seq_len"] == 1024


def test_model_validation_rejects_deep_thin_architecture_regressions() -> None:
    base_354m = load_yaml(CONFIG_DIR / "model_354m.yaml")
    with pytest.raises(ConfigError, match="hidden_size must be 768"):
        parse_config({**base_354m, "hidden_size": 1152, "head_dim": 96})

    with pytest.raises(ConfigError, match="n_query_heads \\* head_dim"):
        parse_config({**base_354m, "head_dim": 32})

    base_480m = load_yaml(CONFIG_DIR / "model_480m.yaml")
    with pytest.raises(ConfigError, match="64 layers"):
        parse_config({**base_480m, "n_layers": 63})


def test_real_tpu_1024_train_configs_include_required_hardening_fields() -> None:
    required_keys = {
        "model_config_path",
        "tokenizer_id",
        "training_stage",
        "max_seq_len",
        "prepacked_shards_path",
        "prepacked_manifest_path",
        "shard_manifest_hash_expected",
        "fixed_batch_size",
        "drop_last",
        "dynamic_padding",
        "optimizer",
        "schedule_type",
        "learning_rate",
        "precision",
        "max_grad_norm",
        "gradient_checkpointing",
        "torch_compile",
        "dataloader_num_workers",
        "prefetch_factor",
        "save_every_n_minutes",
        "session_time_limit_minutes",
        "save_before_exit_minutes",
        "checkpoint_interval_steps",
        "eval_interval_steps",
        "platform",
        "trainer_name",
        "checkpoint_dir",
        "durable_checkpoint_store",
        "data_snapshot",
        "tpu_data_cursor",
    }
    for filename in TPU_1024_CONFIGS:
        config = validate_tpu_train_config(CONFIG_DIR / filename)

        assert required_keys.issubset(config)
        assert config["max_seq_len"] == 1024
        assert config["platform"] == "kaggle-tpu-background"
        assert config["drop_last"] is True
        assert config["dynamic_padding"] is False
        assert config["optimizer"] == "adamw"
        assert config["schedule_type"] == "wsd"
        assert config["precision"] == "bf16"
        assert config["torch_compile"] is False
        assert config["tpu_data_cursor"]["enabled"] is True


def test_2048_configs_are_later_stage_only() -> None:
    train_config = load_config(CONFIG_DIR / "train_tpu_354m_2048_later.yaml")
    prepack_config = load_config(CONFIG_DIR / "prepack_tpu_2048_later.yaml")

    assert train_config["max_seq_len"] == 2048
    assert train_config["later_stage_only"] is True
    assert train_config["stage_gate"] == "after_1024_benchmarking"
    assert prepack_config["later_stage_only"] is True
    assert prepack_config["stage_gate"] == "after_1024_benchmarking"


def test_all_train_configs_declare_data_snapshot_policy() -> None:
    for filename in CONFIG_DIR.glob("train_*.yaml"):
        validate_train_config_data_snapshot(load_config(filename))


def test_dataset_config_marks_structured_synthetic_as_offline_prepack_source() -> None:
    datasets = load_config(CONFIG_DIR / "datasets.yaml")
    synthetic = datasets["sources"]["synthetic"]

    assert synthetic["local_preview_only"] is True
    assert synthetic["generated_offline_before_prepack"] is True
    assert synthetic["parent_teacher_calls_in_tpu_loop"] is False
