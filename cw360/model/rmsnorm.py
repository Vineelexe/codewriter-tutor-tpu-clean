from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root mean square normalization for pre-norm decoder blocks."""

    def __init__(self, hidden_size: int, eps: float = 1.0e-5) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        variance = normalized.pow(2).mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.eps)
        return self.weight.to(dtype=input_dtype) * normalized.to(dtype=input_dtype)
