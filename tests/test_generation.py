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


def test_greedy_generation_stops_on_eos_without_cache() -> None:
    config = tiny_lm_config()
    model = CodeWriterTutorLM(config)
    next_tokens = [4, 7]

    def fake_forward(
        self: CodeWriterTutorLM,
        input_ids: torch.Tensor,
        **_: object,
    ) -> CodeWriterTutorLMOutput:
        token = next_tokens[min(input_ids.shape[1] - 2, len(next_tokens) - 1)]
        logits = torch.full((input_ids.shape[0], input_ids.shape[1], config.vocab_size), -10.0)
        logits[:, -1, token] = 10.0
        return CodeWriterTutorLMOutput(logits=logits)

    model.forward = MethodType(fake_forward, model)

    generated = model.generate(
        torch.tensor([[1, 2]], dtype=torch.long),
        max_new_tokens=4,
        eos_token_id=7,
        use_cache=False,
    )

    assert generated.tolist() == [[1, 2, 4, 7]]


def test_temperature_sampling_supports_top_k() -> None:
    model = CodeWriterTutorLM(tiny_lm_config())
    logits = torch.tensor([[0.0, 9.0, 8.0, 7.0]])
    torch.manual_seed(0)

    token = model._sample_next_token(logits, temperature=1.0, top_k=1, top_p=None)

    assert token.tolist() == [1]


def test_temperature_sampling_supports_top_p() -> None:
    model = CodeWriterTutorLM(tiny_lm_config())
    logits = torch.tensor([[10.0, 9.0, 0.0, -1.0]])
    torch.manual_seed(0)

    token = model._sample_next_token(logits, temperature=1.0, top_k=None, top_p=0.70)

    assert token.tolist() == [0]


def test_real_cpu_tiny_generation_runs() -> None:
    model = CodeWriterTutorLM(tiny_lm_config(), init_seed=9)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    generated = model.generate(input_ids, max_new_tokens=2, use_cache=False)

    assert generated.shape == (1, 5)
