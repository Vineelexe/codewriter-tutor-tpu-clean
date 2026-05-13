from __future__ import annotations

import torch
from torch import nn

from cw360.config import ModelConfig
from cw360.model.attention import AttentionOutput, GroupedQueryAttention, KVCache
from cw360.model.mlp import SwiGLUMLP
from cw360.model.rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    """Pre-norm decoder Transformer block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.self_attn = GroupedQueryAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: KVCache | None = None,
        use_cache: bool = False,
    ) -> AttentionOutput:
        residual = hidden_states
        attention_result = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = residual + attention_result.hidden_states

        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return AttentionOutput(
            hidden_states=hidden_states,
            past_key_values=attention_result.past_key_values,
        )
