from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from cw360.config import ModelConfig
from cw360.constants import VALID_MODEL_SIZE_LABELS

EXPECTED_PARAMETER_COUNTS: dict[str, int] = {
    "354m": 354_000_000,
    "420m": 423_000_000,
    "480m": 479_000_000,
}


@dataclass(frozen=True)
class ParameterCountReport:
    size_label: str
    total_parameters: int
    trainable_parameters: int
    embedding_parameters: int
    per_block_estimate: int
    expected_parameters: int | None
    actual_parameters: int | None = None
    size_label_valid: bool = True

    @property
    def delta_from_expected(self) -> int | None:
        if self.expected_parameters is None:
            return None
        return self.total_parameters - self.expected_parameters

    @property
    def relative_delta_from_expected(self) -> float | None:
        if self.expected_parameters is None:
            return None
        return self.delta_from_expected / self.expected_parameters


def estimate_embedding_parameters(config: ModelConfig) -> int:
    return config.vocab_size * config.hidden_size


def estimate_block_parameters(config: ModelConfig) -> int:
    head_dim = config.hidden_size // config.num_attention_heads
    kv_size = config.num_key_value_heads * head_dim

    q_proj = config.hidden_size * config.hidden_size
    k_proj = config.hidden_size * kv_size
    v_proj = config.hidden_size * kv_size
    if config.qkv_bias:
        q_proj += config.hidden_size
        k_proj += kv_size
        v_proj += kv_size
    o_proj = config.hidden_size * config.hidden_size
    if config.use_bias:
        o_proj += config.hidden_size

    mlp = 3 * config.hidden_size * config.intermediate_size
    if config.use_bias:
        mlp += (2 * config.intermediate_size) + config.hidden_size

    norms = 2 * config.hidden_size
    return q_proj + k_proj + v_proj + o_proj + mlp + norms


def estimate_total_parameters(config: ModelConfig) -> int:
    embedding_parameters = estimate_embedding_parameters(config)
    block_parameters = config.num_hidden_layers * estimate_block_parameters(config)
    final_norm_parameters = config.hidden_size
    return embedding_parameters + block_parameters + final_norm_parameters


def count_unique_parameters(model: nn.Module, *, trainable_only: bool = False) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in model.parameters():
        if trainable_only and not parameter.requires_grad:
            continue
        data_ptr = parameter.untyped_storage().data_ptr()
        if data_ptr in seen:
            continue
        seen.add(data_ptr)
        total += parameter.numel()
    return total


def build_parameter_report(
    config: ModelConfig,
    *,
    model: nn.Module | None = None,
) -> ParameterCountReport:
    if config.size_label not in VALID_MODEL_SIZE_LABELS:
        raise ValueError(f"invalid size_label: {config.size_label}")

    expected_total = estimate_total_parameters(config)
    actual_total = count_unique_parameters(model) if model is not None else None
    trainable_total = (
        count_unique_parameters(model, trainable_only=True) if model is not None else expected_total
    )
    return ParameterCountReport(
        size_label=config.size_label,
        total_parameters=actual_total if actual_total is not None else expected_total,
        trainable_parameters=trainable_total,
        embedding_parameters=estimate_embedding_parameters(config),
        per_block_estimate=estimate_block_parameters(config),
        expected_parameters=EXPECTED_PARAMETER_COUNTS.get(config.size_label),
        actual_parameters=actual_total,
        size_label_valid=config.size_label in VALID_MODEL_SIZE_LABELS,
    )


def validate_size_label_count(report: ParameterCountReport, *, tolerance: float = 0.01) -> bool:
    if report.expected_parameters is None:
        return report.size_label == "tiny"
    delta = abs(report.total_parameters - report.expected_parameters)
    return delta <= report.expected_parameters * tolerance
