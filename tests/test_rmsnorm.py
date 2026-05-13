from __future__ import annotations

import torch

from cw360.model import RMSNorm


def test_rmsnorm_shape_and_gradient() -> None:
    norm = RMSNorm(hidden_size=8, eps=1.0e-5)
    hidden_states = torch.randn(2, 3, 8, requires_grad=True)

    output = norm(hidden_states)
    assert output.shape == hidden_states.shape

    output.sum().backward()
    assert hidden_states.grad is not None
    assert norm.weight.grad is not None


def test_rmsnorm_preserves_input_dtype_under_bfloat16() -> None:
    norm = RMSNorm(hidden_size=8, eps=1.0e-5)
    hidden_states = torch.randn(2, 3, 8, dtype=torch.bfloat16)

    output = norm(hidden_states)

    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output.float()).all()
