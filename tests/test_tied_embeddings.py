from __future__ import annotations

from cw360.config import ModelConfig
from cw360.model import CodeWriterTutorLM


def tiny_lm_config() -> ModelConfig:
    return ModelConfig.model_validate(
        {
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
    )


def test_lm_head_weight_shares_storage_with_token_embedding() -> None:
    model = CodeWriterTutorLM(tiny_lm_config())

    assert model.lm_head.weight is model.token_embedding.weight
    assert (
        model.lm_head.weight.untyped_storage().data_ptr()
        == model.token_embedding.weight.untyped_storage().data_ptr()
    )
