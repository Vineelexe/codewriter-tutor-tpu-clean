from __future__ import annotations

from types import MethodType

import torch

from cw360.config import ModelConfig
from cw360.model import CodeWriterTutorLM, CodeWriterTutorLMOutput


def tiny_lm_config() -> ModelConfig:
    return ModelConfig.model_validate(
        {
            "config_type": "model",
            "size_label": "tiny",
            "vocab_size": 16,
            "max_position_embeddings": 32,
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 32,
        }
    )


def fake_cache(config: ModelConfig) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    key = torch.zeros(1, config.num_key_value_heads, 1, config.head_dim)
    value = torch.zeros(1, config.num_key_value_heads, 1, config.head_dim)
    return tuple((key, value) for _ in range(config.num_hidden_layers))


def test_generation_uses_single_token_steps_after_initial_cached_context() -> None:
    config = tiny_lm_config()
    model = CodeWriterTutorLM(config)
    seen_lengths: list[int] = []
    seen_cache_flags: list[bool] = []

    def fake_forward(
        self: CodeWriterTutorLM,
        input_ids: torch.Tensor,
        **kwargs: object,
    ) -> CodeWriterTutorLMOutput:
        seen_lengths.append(input_ids.shape[1])
        seen_cache_flags.append(kwargs.get("past_key_values") is not None)
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], config.vocab_size)
        logits[:, -1, 3] = 1.0
        return CodeWriterTutorLMOutput(logits=logits, past_key_values=fake_cache(config))

    model.forward = MethodType(fake_forward, model)

    generated = model.generate(
        torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        max_new_tokens=3,
        use_cache=True,
    )

    assert generated.shape == (1, 7)
    assert seen_lengths == [4, 1, 1]
    assert seen_cache_flags == [False, True, True]


def test_generation_no_cache_fallback_recomputes_context_for_debugging() -> None:
    config = tiny_lm_config()
    model = CodeWriterTutorLM(config)
    seen_lengths: list[int] = []

    def fake_forward(
        self: CodeWriterTutorLM,
        input_ids: torch.Tensor,
        **_: object,
    ) -> CodeWriterTutorLMOutput:
        seen_lengths.append(input_ids.shape[1])
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], config.vocab_size)
        logits[:, -1, 3] = 1.0
        return CodeWriterTutorLMOutput(logits=logits)

    model.forward = MethodType(fake_forward, model)

    model.generate(
        torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        max_new_tokens=3,
        use_cache=False,
    )

    assert seen_lengths == [4, 5, 6]
