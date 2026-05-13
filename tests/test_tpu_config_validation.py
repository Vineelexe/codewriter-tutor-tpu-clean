from __future__ import annotations

from pathlib import Path

import pytest

from cw360.config import ConfigError, ModelConfig, load_config, parse_config
from cw360.prepack.config import PrepackConfigError, parse_prepack_config, validate_prepack_config
from cw360.train.validate import (
    validate_teacher_candidates_config,
    validate_tpu_train_config,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_tpu_validator_accepts_all_real_1024_candidate_configs() -> None:
    for filename in (
        "train_tpu_354m_1024.yaml",
        "train_tpu_420m_1024.yaml",
        "train_tpu_480m_1024.yaml",
    ):
        config = validate_tpu_train_config(CONFIG_DIR / filename)
        assert config["max_seq_len"] == 1024


def test_model_validation_rejects_required_architecture_regressions() -> None:
    base = load_config(CONFIG_DIR / "model_354m.yaml")
    with pytest.raises(ConfigError, match="qkv_bias"):
        parse_config({**base, "qkv_bias": False})
    with pytest.raises(ConfigError, match="tie_word_embeddings"):
        parse_config({**base, "tie_word_embeddings": False})
    with pytest.raises(ConfigError, match="mlp_hidden_size"):
        parse_config({**base, "mlp_hidden_size": base["intermediate_size"] + 1})


def test_tpu_validation_rejects_cuda_fp16_and_dynamic_padding() -> None:
    config = load_config(CONFIG_DIR / "train_tpu_354m_1024.yaml")

    with pytest.raises(ValueError, match="bf16"):
        validate_tpu_train_config({**config, "precision": "fp16"})
    with pytest.raises(ValueError, match="GradScaler"):
        validate_tpu_train_config({**config, "use_fp16_grad_scaler": True})
    with pytest.raises(ValueError, match="dynamic padding"):
        validate_tpu_train_config({**config, "dynamic_padding": True})
    with pytest.raises(ValueError, match="drop_last"):
        validate_tpu_train_config({**config, "drop_last": False})


def test_teacher_candidates_task_mix_sums_to_one_and_includes_refactor() -> None:
    config = load_config(CONFIG_DIR / "teacher_candidates.yaml")
    validate_teacher_candidates_config(config)

    bad_sum = {**config, "task_mix": {**config["task_mix"], "refactor": 0.01}}
    with pytest.raises(ValueError, match="sum to 1.0"):
        validate_teacher_candidates_config(bad_sum)

    no_refactor = dict(config["task_mix"])
    no_refactor.pop("refactor")
    with pytest.raises(ValueError, match="refactor"):
        validate_teacher_candidates_config({**config, "task_mix": no_refactor})


def test_prepack_dataset_mixture_must_sum_to_one() -> None:
    config = parse_prepack_config(load_config(CONFIG_DIR / "prepack_tpu_1024.yaml"))
    validate_prepack_config(config)

    bad = load_config(CONFIG_DIR / "prepack_tpu_1024.yaml")
    bad["mixture"] = {**bad["mixture"], "fineweb_edu": 0.01}
    with pytest.raises(PrepackConfigError, match="sum to 1.0"):
        validate_prepack_config(parse_prepack_config(bad))


def test_tokenizer_vocab_limit_is_config_validated() -> None:
    with pytest.raises(ValueError, match="uint16"):
        ModelConfig.model_validate(
            {
                "config_type": "model",
                "size_label": "tiny",
                "tokenizer_name": "bigcode/starcoder2-15b",
                "vocab_size": 65_536,
                "max_position_embeddings": 128,
                "hidden_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "intermediate_size": 256,
                "mlp_hidden_size": 256,
                "qkv_bias": True,
                "tie_word_embeddings": True,
            }
        )
