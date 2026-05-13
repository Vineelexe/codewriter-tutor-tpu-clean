from __future__ import annotations

from cw360.config import ModelConfig
from cw360.model import CodeWriterTutorLM
from cw360.model.count import (
    build_parameter_report,
    estimate_block_parameters,
    estimate_embedding_parameters,
    estimate_total_parameters,
    validate_size_label_count,
)


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


def test_parameter_count_matches_actual_tiny_model_with_tied_head() -> None:
    config = tiny_lm_config()
    model = CodeWriterTutorLM(config)
    report = build_parameter_report(config, model=model)

    assert report.actual_parameters == estimate_total_parameters(config)
    assert report.total_parameters == report.trainable_parameters
    assert report.embedding_parameters == estimate_embedding_parameters(config)
    assert report.per_block_estimate == estimate_block_parameters(config)


def test_tiny_size_label_is_valid_without_large_expected_target() -> None:
    report = build_parameter_report(tiny_lm_config())

    assert report.size_label_valid
    assert report.expected_parameters is None
    assert validate_size_label_count(report)
