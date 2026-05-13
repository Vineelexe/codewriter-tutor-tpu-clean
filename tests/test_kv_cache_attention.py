from __future__ import annotations

import torch

from cw360.config import ModelConfig
from cw360.model import GroupedQueryAttention


def tiny_config() -> ModelConfig:
    return ModelConfig.model_validate(
        {
            "config_type": "model",
            "size_label": "tiny",
            "max_position_embeddings": 32,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 64,
        }
    )


def test_attention_cache_shapes_grow_with_incremental_decode() -> None:
    attention = GroupedQueryAttention(tiny_config())
    first_tokens = torch.randn(2, 3, 32)

    first = attention(first_tokens, use_cache=True)

    assert first.hidden_states.shape == (2, 3, 32)
    assert first.past_key_values is not None
    assert first.past_key_values[0].shape == (2, 2, 3, 8)
    assert first.past_key_values[1].shape == (2, 2, 3, 8)

    next_token = torch.randn(2, 1, 32)
    second = attention(next_token, past_key_values=first.past_key_values, use_cache=True)

    assert second.hidden_states.shape == (2, 1, 32)
    assert second.past_key_values is not None
    assert second.past_key_values[0].shape == (2, 2, 4, 8)
    assert second.past_key_values[1].shape == (2, 2, 4, 8)


def test_attention_without_cache_returns_no_cache() -> None:
    attention = GroupedQueryAttention(tiny_config())
    hidden_states = torch.randn(2, 3, 32)

    output = attention(hidden_states, use_cache=False)

    assert output.hidden_states.shape == hidden_states.shape
    assert output.past_key_values is None
