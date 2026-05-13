from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from cw360.config import ModelConfig


class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward network used inside decoder blocks."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.activation not in {"silu", "swiglu"}:
            raise ValueError(f"unsupported activation for SwiGLUMLP: {config.activation}")
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.use_bias,
        )
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.use_bias)
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.use_bias,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.down_proj(gated)
