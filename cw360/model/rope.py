from __future__ import annotations

import torch
from torch import nn


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    half = hidden_states.shape[-1] // 2
    first_half = hidden_states[..., :half]
    second_half = hidden_states[..., half:]
    return torch.cat((-second_half, first_half), dim=-1)


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding cache for attention tensors shaped [B, H, T, D]."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 1_000_000.0) -> None:
        super().__init__()
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if theta <= 0.0:
            raise ValueError("theta must be positive")

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("_cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("_sin_cached", torch.empty(0), persistent=False)
        self._cache_seq_len = 0

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"requested RoPE sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq.to(device=device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos_cached = emb.cos().to(dtype=dtype)
        self._sin_cached = emb.sin().to(dtype=dtype)
        self._cache_seq_len = seq_len

    def _cos_sin(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        position_ids: torch.Tensor | None,
        position_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_offset < 0:
            raise ValueError("position_offset must be non-negative")

        cache_len = seq_len + position_offset
        cache_is_stale = (
            self._cache_seq_len < cache_len
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
        )
        if cache_is_stale:
            self._build_cache(cache_len, device, dtype)

        if position_ids is not None:
            if position_ids.shape[-1] != seq_len:
                raise ValueError("position_ids length must match query/key sequence length")
            positions = position_ids.to(device=device, dtype=torch.long)
            cos = self._cos_cached.index_select(0, positions.reshape(-1)).reshape(
                *positions.shape, self.head_dim
            )
            sin = self._sin_cached.index_select(0, positions.reshape(-1)).reshape(
                *positions.shape, self.head_dim
            )
            return cos.unsqueeze(1), sin.unsqueeze(1)

        cos = self._cos_cached[position_offset:cache_len].unsqueeze(0).unsqueeze(0)
        sin = self._sin_cached[position_offset:cache_len].unsqueeze(0).unsqueeze(0)
        return cos, sin

    def apply_rope(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.shape[-1] != self.head_dim or key.shape[-1] != self.head_dim:
            raise ValueError("query and key last dimension must match RoPE head_dim")
        if query.shape[-2] != key.shape[-2]:
            raise ValueError("query and key sequence length must match before cache concat")

        seq_len = query.shape[-2]
        cos, sin = self._cos_sin(seq_len, query.device, query.dtype, position_ids, position_offset)
        rotated_query = (query * cos) + (_rotate_half(query) * sin)
        rotated_key = (key * cos) + (_rotate_half(key) * sin)
        return rotated_query, rotated_key
