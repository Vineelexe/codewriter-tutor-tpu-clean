from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from cw360.config import ModelConfig
from cw360.model.attention import KVCache
from cw360.model.block import TransformerBlock
from cw360.model.init import initialize_module
from cw360.model.rmsnorm import RMSNorm

PastKeyValues = tuple[KVCache | None, ...]


@dataclass(frozen=True)
class CodeWriterTutorLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    past_key_values: PastKeyValues | None = None


class CodeWriterTutorLM(nn.Module):
    """Decoder-only causal LM specialized for Python code writing and tutoring."""

    def __init__(self, config: ModelConfig, *, init_seed: int = 1234) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        )
        self.final_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.tie_weights()
        self.init_seed = init_seed
        self.initialize_weights(init_seed)
        self.tie_weights()

    def tie_weights(self) -> None:
        self.lm_head.weight = self.token_embedding.weight

    def initialize_weights(self, seed: int) -> None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.apply(initialize_module)

    def _validate_past_key_values(
        self,
        past_key_values: Sequence[KVCache | None] | None,
    ) -> tuple[KVCache | None, ...]:
        if past_key_values is None:
            return tuple(None for _ in self.blocks)
        if len(past_key_values) != len(self.blocks):
            raise ValueError("past_key_values length must match num_hidden_layers")
        return tuple(past_key_values)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        past_key_values: Sequence[KVCache | None] | None = None,
        use_cache: bool = False,
    ) -> CodeWriterTutorLMOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels must have the same shape as input_ids")

        layer_past_key_values = self._validate_past_key_values(past_key_values)
        hidden_states = self.token_embedding(input_ids)
        present_key_values: list[KVCache | None] = []

        for block, layer_past in zip(self.blocks, layer_past_key_values, strict=True):
            block_output = block(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=layer_past,
                use_cache=use_cache,
            )
            hidden_states = block_output.hidden_states
            if use_cache:
                present_key_values.append(block_output.past_key_values)

        logits = self.lm_head(self.final_norm(hidden_states))
        loss = self._causal_lm_loss(logits, labels) if labels is not None else None
        return CodeWriterTutorLMOutput(
            logits=logits,
            loss=loss,
            past_key_values=tuple(present_key_values) if use_cache else None,
        )

    def _causal_lm_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] < 2:
            raise ValueError("causal LM loss requires at least two tokens")
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, self.config.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None,
        use_cache: bool = True,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")
        if input_ids.shape[1] == 0:
            raise ValueError("input_ids must contain at least one token")
        if max_new_tokens == 0:
            return input_ids

        generated = input_ids
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        past_key_values: PastKeyValues | None = None
        cache_attention_mask = attention_mask
        next_input: torch.Tensor | None = None

        for step in range(max_new_tokens):
            if use_cache:
                model_input = generated if step == 0 else next_input
                if model_input is None:
                    raise RuntimeError("internal generation state lost next token")
                output = self(
                    model_input,
                    attention_mask=cache_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = output.past_key_values
            else:
                output = self(generated, attention_mask=attention_mask, use_cache=False)

            next_token_logits = output.logits[:, -1, :]
            next_token = self._sample_next_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            if eos_token_id is not None:
                eos_tensor = torch.full_like(next_token, eos_token_id)
                next_token = torch.where(finished, eos_tensor, next_token)
                finished = finished | next_token.eq(eos_token_id)

            generated = torch.cat((generated, next_token[:, None]), dim=1)
            next_input = next_token[:, None]
            if cache_attention_mask is not None:
                new_mask = torch.ones(
                    (cache_attention_mask.shape[0], 1),
                    dtype=cache_attention_mask.dtype,
                    device=cache_attention_mask.device,
                )
                cache_attention_mask = torch.cat((cache_attention_mask, new_mask), dim=1)
            if eos_token_id is not None and bool(finished.all()):
                break

        return generated

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        *,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
    ) -> torch.Tensor:
        if temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive when provided")
        if top_p is not None and not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1] when provided")
        if temperature == 0.0:
            return torch.argmax(logits, dim=-1)

        filtered_logits = logits / temperature
        if top_k is not None:
            k = min(top_k, filtered_logits.shape[-1])
            kth_values = torch.topk(filtered_logits, k=k, dim=-1).values[:, -1, None]
            filtered_logits = filtered_logits.masked_fill(
                filtered_logits < kth_values,
                float("-inf"),
            )
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = sorted_probs.cumsum(dim=-1)
            remove_sorted = cumulative_probs > top_p
            remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
            remove_sorted[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(remove_sorted, float("-inf"))
            filtered_logits = torch.full_like(filtered_logits, float("-inf"))
            filtered_logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

        probabilities = torch.softmax(filtered_logits, dim=-1)
        return torch.multinomial(probabilities, num_samples=1).squeeze(-1)
