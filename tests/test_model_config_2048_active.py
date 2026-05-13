from __future__ import annotations

from pathlib import Path

from cw360.config import load_model_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_active_354m_2048_model_config_loads_without_architecture_change() -> None:
    config = load_model_config(CONFIG_DIR / "model_354m_2048.yaml")

    assert config.size_label == "354m"
    assert config.tokenizer_name == "bigcode/starcoder2-15b"
    assert config.max_position_embeddings == 2048
    assert config.hidden_size == 768
    assert config.num_hidden_layers == 46
    assert config.num_attention_heads == 12
    assert config.num_key_value_heads == 4
    assert config.head_dim == 64
    assert config.mlp_hidden_size == 2304
    assert config.activation == "swiglu"
