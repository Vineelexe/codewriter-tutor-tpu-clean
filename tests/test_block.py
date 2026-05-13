from __future__ import annotations

import torch

from cw360.config import ModelConfig
from cw360.model import TransformerBlock


def test_transformer_block_shape_gradients_and_cache() -> None:
    config = ModelConfig.model_validate(
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
    block = TransformerBlock(config)
    hidden_states = torch.randn(2, 5, 32, requires_grad=True)

    output = block(hidden_states, use_cache=True)

    assert output.hidden_states.shape == hidden_states.shape
    assert output.past_key_values is not None
    assert output.past_key_values[0].shape == (2, 2, 5, 8)
    output.hidden_states.sum().backward()
    assert hidden_states.grad is not None
