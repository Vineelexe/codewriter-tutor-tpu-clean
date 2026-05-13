from __future__ import annotations

import pytest
import torch

from cw360.model import RotaryEmbedding


def test_rope_shape_and_dtype_movement() -> None:
    rope = RotaryEmbedding(head_dim=8, max_seq_len=16)
    query = torch.randn(2, 4, 5, 8, dtype=torch.float32)
    key = torch.randn(2, 2, 5, 8, dtype=torch.float32)

    rotated_query, rotated_key = rope.apply_rope(query, key)

    assert rotated_query.shape == query.shape
    assert rotated_key.shape == key.shape
    assert rotated_query.dtype == query.dtype
    assert rope._cos_cached.device == query.device


def test_rope_supports_bfloat16_inputs() -> None:
    rope = RotaryEmbedding(head_dim=8, max_seq_len=16)
    query = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)
    key = torch.randn(1, 1, 4, 8, dtype=torch.bfloat16)

    rotated_query, rotated_key = rope.apply_rope(query, key)

    assert rotated_query.dtype == torch.bfloat16
    assert rotated_key.dtype == torch.bfloat16


def test_rope_rejects_odd_head_dim() -> None:
    with pytest.raises(ValueError, match="even"):
        RotaryEmbedding(head_dim=7, max_seq_len=16)


def test_rope_rejects_sequences_beyond_configured_max() -> None:
    rope = RotaryEmbedding(head_dim=8, max_seq_len=3)
    query = torch.randn(1, 1, 4, 8)
    key = torch.randn(1, 1, 4, 8)

    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        rope.apply_rope(query, key)
