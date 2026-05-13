from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import TypeAlias

import torch
from torch import nn
from torch.nn import functional as F

from cw360.config import ModelConfig
from cw360.model.rope import RotaryEmbedding

KVCache: TypeAlias = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class AttentionOutput:
    hidden_states: torch.Tensor
    past_key_values: KVCache | None = None


class GroupedQueryAttention(nn.Module):
    """Decoder-only grouped-query self-attention with optional inference KV cache."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.hidden_size != config.num_attention_heads * config.head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        if config.num_attention_heads % config.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        self.hidden_size = config.hidden_size
        self.num_query_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.gqa_group_size = config.gqa_group_size
        self.attention_dropout = config.attention_dropout

        kv_size = self.num_key_value_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.k_proj = nn.Linear(config.hidden_size, kv_size, bias=config.qkv_bias)
        self.v_proj = nn.Linear(config.hidden_size, kv_size, bias=config.qkv_bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        self.rope = RotaryEmbedding(
            head_dim=self.head_dim,
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
        )

    def _shape(self, tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(batch_size, seq_len, num_heads, self.head_dim).transpose(1, 2)

    def _repeat_key_value(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.gqa_group_size == 1:
            return tensor
        return tensor.repeat_interleave(self.gqa_group_size, dim=1)

    def _causal_mask(
        self,
        query_len: int,
        key_len: int,
        past_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        query_positions = torch.arange(query_len, device=device).unsqueeze(1) + past_len
        key_positions = torch.arange(key_len, device=device).unsqueeze(0)
        return key_positions <= query_positions

    def _combine_masks(
        self,
        causal_mask: torch.Tensor,
        attention_mask: torch.Tensor | None,
        batch_size: int,
        key_len: int,
    ) -> torch.Tensor:
        mask = causal_mask.unsqueeze(0).unsqueeze(0)
        if attention_mask is None:
            return mask

        if attention_mask.ndim == 2:
            if attention_mask.shape != (batch_size, key_len):
                raise ValueError("2D attention_mask must have shape [batch, key_len]")
            padding_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
            return mask & padding_mask
        if attention_mask.ndim == 4:
            return mask & attention_mask.to(dtype=torch.bool)
        raise ValueError("attention_mask must be 2D or 4D when provided")

    def _manual_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) / sqrt(self.head_dim)
        scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores, dim=-1).to(dtype=query.dtype)
        if self.training and self.attention_dropout > 0.0:
            probabilities = F.dropout(probabilities, p=self.attention_dropout)
        return torch.matmul(probabilities, value)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: KVCache | None = None,
        use_cache: bool = False,
    ) -> AttentionOutput:
        batch_size, query_len, hidden_size = hidden_states.shape
        if hidden_size != self.hidden_size:
            raise ValueError(f"expected hidden_size={self.hidden_size}, got {hidden_size}")

        query = self._shape(self.q_proj(hidden_states), self.num_query_heads)
        key = self._shape(self.k_proj(hidden_states), self.num_key_value_heads)
        value = self._shape(self.v_proj(hidden_states), self.num_key_value_heads)

        past_len = 0
        if past_key_values is not None:
            past_key, past_value = past_key_values
            expected_cache_shape = (batch_size, self.num_key_value_heads, self.head_dim)
            if (
                past_key.shape[0] != expected_cache_shape[0]
                or past_key.shape[1] != expected_cache_shape[1]
            ):
                raise ValueError(
                    "past_key cache batch/head dimensions do not match attention config"
                )
            if past_value.shape[:2] != past_key.shape[:2] or past_value.shape[-1] != self.head_dim:
                raise ValueError("past_value cache dimensions do not match past_key")
            past_len = past_key.shape[-2]
            key_offset = past_len
        else:
            key_offset = 0

        query, key = self.rope.apply_rope(
            query,
            key,
            position_ids=position_ids,
            position_offset=key_offset,
        )

        if past_key_values is not None:
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        present_key_values = (key, value) if use_cache else None
        key_len = key.shape[-2]
        repeated_key = self._repeat_key_value(key)
        repeated_value = self._repeat_key_value(value)
        causal_mask = self._causal_mask(query_len, key_len, past_len, hidden_states.device)
        combined_mask = self._combine_masks(causal_mask, attention_mask, batch_size, key_len)

        try:
            attention_output = F.scaled_dot_product_attention(
                query,
                repeated_key,
                repeated_value,
                attn_mask=combined_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=False,
            )
        except (AttributeError, NotImplementedError, RuntimeError):
            attention_output = self._manual_attention(
                query,
                repeated_key,
                repeated_value,
                combined_mask,
            )

        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, query_len, self.hidden_size
        )
        return AttentionOutput(
            hidden_states=self.o_proj(attention_output),
            past_key_values=present_key_values,
        )
