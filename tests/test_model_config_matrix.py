from __future__ import annotations

from pathlib import Path

from cw360.config import load_model_config
from cw360.model import GroupedQueryAttention, RMSNorm, RotaryEmbedding, SwiGLUMLP, TransformerBlock

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_target_model_configs_instantiate_core_modules_without_forward() -> None:
    for filename in ["model_354m.yaml", "model_420m.yaml", "model_480m.yaml", "model_tiny.yaml"]:
        config = load_model_config(CONFIG_DIR / filename)

        RMSNorm(config.hidden_size, eps=config.norm_eps)
        RotaryEmbedding(
            head_dim=config.head_dim,
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
        )
        GroupedQueryAttention(config)
        SwiGLUMLP(config)
        TransformerBlock(config)

        assert config.vocab_size == 49152
        assert config.max_position_embeddings >= 512
