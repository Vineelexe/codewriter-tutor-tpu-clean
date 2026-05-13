from __future__ import annotations

from cw360.config import ModelConfig
from cw360.local.memory import (
    SPEED_BOTTLENECK_WARNING,
    build_local_memory_report,
    build_memory_estimates,
)


def _config() -> ModelConfig:
    return ModelConfig.model_validate(
        {
            "config_type": "model",
            "size_label": "tiny",
            "tokenizer_name": "bigcode/starcoder2-15b",
            "vocab_size": 32,
            "max_position_embeddings": 32,
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 32,
        }
    )


def test_memory_estimates_are_deterministic() -> None:
    estimates = {estimate.precision: estimate for estimate in build_memory_estimates(100)}

    assert estimates["fp32"].parameter_bytes == 400
    assert estimates["bf16_fp16"].parameter_bytes == 200
    assert estimates["int8"].parameter_bytes == 100
    assert estimates["4bit_theoretical"].parameter_bytes == 50
    assert estimates["4bit_theoretical"].theoretical


def test_local_memory_report_is_structured() -> None:
    report = build_local_memory_report(_config(), available_ram_bytes=16 * 1024**3)
    payload = report.to_dict()

    assert payload["available_ram_bytes"] == 16 * 1024**3
    assert payload["available_ram_gib"] == 16
    assert payload["total_parameters"] > 0
    assert payload["warning"] == SPEED_BOTTLENECK_WARNING
    assert {item["precision"] for item in payload["estimates"]} == {
        "fp32",
        "bf16_fp16",
        "int8",
        "4bit_theoretical",
    }
