from __future__ import annotations

import torch

from cw360.config import ModelConfig
from cw360.model import SwiGLUMLP


def test_swiglu_mlp_shape_and_gradients() -> None:
    config = ModelConfig.model_validate(
        {
            "config_type": "model",
            "size_label": "tiny",
            "max_position_embeddings": 32,
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 40,
        }
    )
    mlp = SwiGLUMLP(config)
    hidden_states = torch.randn(2, 3, 16, requires_grad=True)

    output = mlp(hidden_states)

    assert output.shape == hidden_states.shape
    output.sum().backward()
    assert hidden_states.grad is not None
    assert mlp.gate_proj.weight.grad is not None
