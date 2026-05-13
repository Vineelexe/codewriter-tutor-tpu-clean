from __future__ import annotations

import pytest
import torch

from cw360.config import ModelConfig
from cw360.model import GroupedQueryAttention


def tiny_config(**overrides: object) -> ModelConfig:
    data = {
        "config_type": "model",
        "size_label": "tiny",
        "max_position_embeddings": 32,
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 64,
    }
    data.update(overrides)
    return ModelConfig.model_validate(data)


def test_attention_output_shape_and_gradients() -> None:
    attention = GroupedQueryAttention(tiny_config())
    hidden_states = torch.randn(2, 5, 32, requires_grad=True)

    output = attention(hidden_states).hidden_states

    assert output.shape == hidden_states.shape
    output.sum().backward()
    assert hidden_states.grad is not None
    assert attention.q_proj.weight.grad is not None


def test_attention_uses_grouped_query_shapes() -> None:
    attention = GroupedQueryAttention(tiny_config())
    hidden_states = torch.randn(2, 5, 32)

    query = attention._shape(attention.q_proj(hidden_states), attention.num_query_heads)
    key = attention._shape(attention.k_proj(hidden_states), attention.num_key_value_heads)
    repeated_key = attention._repeat_key_value(key)

    assert query.shape == (2, 4, 5, 8)
    assert key.shape == (2, 2, 5, 8)
    assert repeated_key.shape == query.shape


def test_attention_rejects_invalid_hidden_size_at_forward() -> None:
    attention = GroupedQueryAttention(tiny_config())

    with pytest.raises(ValueError, match="expected hidden_size"):
        attention(torch.randn(2, 5, 31))


def test_attention_rejects_invalid_grouping_clearly() -> None:
    with pytest.raises(ValueError, match="num_attention_heads must be divisible"):
        ModelConfig.model_validate(
            {
                "config_type": "model",
                "size_label": "tiny",
                "max_position_embeddings": 32,
                "hidden_size": 48,
                "num_hidden_layers": 2,
                "num_attention_heads": 6,
                "num_key_value_heads": 4,
                "intermediate_size": 64,
            }
        )


def test_causal_attention_does_not_crash() -> None:
    attention = GroupedQueryAttention(tiny_config())
    hidden_states = torch.randn(1, 8, 32)

    output = attention(hidden_states).hidden_states

    assert output.shape == hidden_states.shape
