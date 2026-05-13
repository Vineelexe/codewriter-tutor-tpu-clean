from __future__ import annotations

import torch
from torch.nn import functional as F

from cw360.config import ModelConfig
from cw360.model import CodeWriterTutorLM


def tiny_lm_config(**overrides: object) -> ModelConfig:
    data = {
        "config_type": "model",
        "size_label": "tiny",
        "vocab_size": 32,
        "max_position_embeddings": 32,
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 32,
    }
    data.update(overrides)
    return ModelConfig.model_validate(data)


def test_lm_forward_logits_and_loss_shape() -> None:
    config = tiny_lm_config()
    model = CodeWriterTutorLM(config, init_seed=7)
    input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)

    output = model(input_ids, labels=input_ids)

    assert output.logits.shape == (2, 4, config.vocab_size)
    assert output.loss is not None
    assert output.past_key_values is None


def test_lm_loss_uses_next_token_shift() -> None:
    config = tiny_lm_config()
    model = CodeWriterTutorLM(config, init_seed=11)
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    output = model(input_ids, labels=input_ids)
    expected = F.cross_entropy(
        output.logits[:, :-1, :].contiguous().view(-1, config.vocab_size),
        input_ids[:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )

    assert output.loss is not None
    assert torch.allclose(output.loss, expected)


def test_lm_initialization_is_deterministic() -> None:
    config = tiny_lm_config()
    first = CodeWriterTutorLM(config, init_seed=123)
    second = CodeWriterTutorLM(config, init_seed=123)

    assert torch.equal(first.token_embedding.weight, second.token_embedding.weight)
    assert torch.equal(
        first.blocks[0].self_attn.q_proj.weight,
        second.blocks[0].self_attn.q_proj.weight,
    )


def test_lm_forward_returns_layer_kv_cache() -> None:
    config = tiny_lm_config()
    model = CodeWriterTutorLM(config, init_seed=5)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    first = model(input_ids, use_cache=True)
    assert first.past_key_values is not None
    assert len(first.past_key_values) == config.num_hidden_layers

    second = model(
        torch.tensor([[4]], dtype=torch.long),
        past_key_values=first.past_key_values,
        use_cache=True,
    )
    assert second.logits.shape == (1, 1, config.vocab_size)
    assert second.past_key_values is not None
    assert second.past_key_values[0] is not None
    assert second.past_key_values[0][0].shape[-2] == 4
