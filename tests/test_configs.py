from __future__ import annotations

from pathlib import Path

import pytest

from cw360.config import ConfigError, load_config, parse_config
from cw360.constants import VALID_PLATFORMS, VALID_TRAINING_STAGES

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_required_configs_exist() -> None:
    required_configs = {
        "model_354m.yaml",
        "model_354m_2048.yaml",
        "model_420m.yaml",
        "model_480m.yaml",
        "model_tiny.yaml",
        "model_tiny_tpu.yaml",
        "train_tiny.yaml",
        "train_tiny_prepacked.yaml",
        "train_tpu_354m_1024.yaml",
        "train_tpu_354m_2048.yaml",
        "train_tpu_420m_1024.yaml",
        "train_tpu_480m_1024.yaml",
        "train_tpu_354m_2048_later.yaml",
        "prepack_tpu_1024.yaml",
        "prepack_tpu_2048.yaml",
        "prepack_tpu_2048_later.yaml",
        "eval.yaml",
        "datasets.yaml",
        "checkpoint.yaml",
        "teacher_candidates.yaml",
        "teacher_generation.yaml",
        "local_runtime.yaml",
        "export.yaml",
    }
    assert {path.name for path in CONFIG_DIR.glob("*.yaml")} == required_configs


def test_all_yaml_configs_load() -> None:
    for yaml_file in CONFIG_DIR.glob("*.yaml"):
        config = load_config(yaml_file)
        assert isinstance(config, dict), f"Config {yaml_file.name} must load as a dict"


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        (
            "model_354m.yaml",
            {
                "model_name": "CodeWriter-Tutor-354M",
                "size_label": "354m",
                "hidden_size": 768,
                "num_hidden_layers": 46,
                "num_attention_heads": 12,
                "num_key_value_heads": 4,
                "intermediate_size": 2304,
                "mlp_hidden_size": 2304,
                "max_position_embeddings": 1024,
                "activation": "swiglu",
                "norm": "rmsnorm",
                "position_encoding": "rope",
                "attention": "gqa",
            },
        ),
        (
            "model_420m.yaml",
            {
                "model_name": "CodeWriter-Tutor-420M",
                "size_label": "420m",
                "hidden_size": 768,
                "num_hidden_layers": 56,
                "num_attention_heads": 12,
                "num_key_value_heads": 4,
                "intermediate_size": 2304,
                "mlp_hidden_size": 2304,
                "max_position_embeddings": 1024,
                "activation": "swiglu",
                "norm": "rmsnorm",
                "position_encoding": "rope",
                "attention": "gqa",
            },
        ),
        (
            "model_480m.yaml",
            {
                "model_name": "CodeWriter-Tutor-480M",
                "size_label": "480m",
                "hidden_size": 768,
                "num_hidden_layers": 64,
                "num_attention_heads": 12,
                "num_key_value_heads": 4,
                "intermediate_size": 2304,
                "mlp_hidden_size": 2304,
                "max_position_embeddings": 1024,
                "activation": "swiglu",
                "norm": "rmsnorm",
                "position_encoding": "rope",
                "attention": "gqa",
            },
        ),
    ],
)
def test_model_matrix_exact_values(filename: str, expected: dict[str, int | str]) -> None:
    config = load_config(CONFIG_DIR / filename)
    for key, value in expected.items():
        assert config[key] == value
    assert config["tokenizer_name"] == "bigcode/starcoder2-15b"


def test_invalid_head_dimension_is_rejected() -> None:
    with pytest.raises(ConfigError, match="hidden_size must be divisible"):
        parse_config(
            {
                "config_type": "model",
                "size_label": "tiny",
                "max_position_embeddings": 128,
                "hidden_size": 130,
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "intermediate_size": 256,
            }
        )


def test_invalid_gqa_grouping_is_rejected() -> None:
    with pytest.raises(ConfigError, match="num_attention_heads must be divisible"):
        parse_config(
            {
                "config_type": "model",
                "size_label": "tiny",
                "max_position_embeddings": 128,
                "hidden_size": 120,
                "num_hidden_layers": 2,
                "num_attention_heads": 6,
                "num_key_value_heads": 4,
                "intermediate_size": 256,
            }
        )


def test_tiny_configs_load() -> None:
    assert load_config(CONFIG_DIR / "model_tiny.yaml")["size_label"] == "tiny"
    assert load_config(CONFIG_DIR / "model_tiny_tpu.yaml")["size_label"] == "tiny"


def test_training_stage_enum_exists() -> None:
    assert VALID_TRAINING_STAGES == {
        "base_pretrain",
        "fim_train",
        "instruction_tune",
        "eval_only",
    }
    assert load_config(CONFIG_DIR / "train_tiny.yaml")["training_stage"] in VALID_TRAINING_STAGES


def test_local_runtime_loads() -> None:
    runtime = load_config(CONFIG_DIR / "local_runtime.yaml")
    assert runtime["platform"] == "local-cpu-test"
    assert runtime["platform"] in VALID_PLATFORMS


def test_tpu_configs_load_without_torch_xla_import() -> None:
    assert load_config(CONFIG_DIR / "model_tiny_tpu.yaml")["max_position_embeddings"] == 1024
    for filename in [
        "train_tpu_354m_1024.yaml",
        "train_tpu_354m_2048.yaml",
        "train_tpu_420m_1024.yaml",
        "train_tpu_480m_1024.yaml",
        "prepack_tpu_1024.yaml",
        "prepack_tpu_2048.yaml",
    ]:
        assert isinstance(load_config(CONFIG_DIR / filename), dict)
