from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from cw360.constants import VALID_MODEL_SIZE_LABELS, VALID_PLATFORMS, VALID_TRAINING_STAGES
from cw360.tokenizer.special_tokens import (
    DEFAULT_TOKENIZER_ID,
    UINT16_TOKEN_ID_LIMIT,
    is_starcoder2_tokenizer_id,
)


class ConfigError(ValueError):
    """Raised when a YAML config cannot be loaded or validated."""


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_type: Literal["model"]
    model_name: str = "cw360"
    size_label: Literal["354m", "420m", "480m", "tiny"]
    tokenizer_name: str = DEFAULT_TOKENIZER_ID
    vocab_size: int = 49152
    max_position_embeddings: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    mlp_hidden_size: int | None = None
    declared_head_dim: int | None = Field(default=None, exclude=True)
    norm: Literal["rmsnorm"] = "rmsnorm"
    activation: Literal["silu", "swiglu"] = "swiglu"
    position_encoding: Literal["rope"] = "rope"
    attention: Literal["gqa"] = "gqa"
    norm_eps: float = 1.0e-5
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = True
    qkv_bias: bool = True
    use_bias: bool = False
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def normalize_architecture_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        normalized = dict(data)
        for alias, target in {
            "max_seq_len": "max_position_embeddings",
            "n_layers": "num_hidden_layers",
            "n_query_heads": "num_attention_heads",
            "n_kv_heads": "num_key_value_heads",
            "tied_embeddings": "tie_word_embeddings",
            "head_dim": "declared_head_dim",
        }.items():
            if alias not in normalized:
                continue
            alias_value = normalized.pop(alias)
            if target in normalized and normalized[target] != alias_value:
                raise ValueError(f"{alias} conflicts with {target}")
            normalized[target] = alias_value

        if "intermediate_size" not in normalized and "mlp_hidden_size" in normalized:
            normalized["intermediate_size"] = normalized["mlp_hidden_size"]
        return normalized

    @field_validator("size_label")
    @classmethod
    def validate_size_label(cls, value: str) -> str:
        if value not in VALID_MODEL_SIZE_LABELS:
            raise ValueError(f"size_label must be one of {sorted(VALID_MODEL_SIZE_LABELS)}")
        return value

    @field_validator(
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "mlp_hidden_size",
        "declared_head_dim",
        "max_position_embeddings",
    )
    @classmethod
    def positive_int(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("architecture dimensions must be positive integers")
        return value

    @field_validator("vocab_size")
    @classmethod
    def validate_vocab_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("vocab_size must be positive")
        if value >= UINT16_TOKEN_ID_LIMIT:
            raise ValueError("tokenizer vocab must fit uint16 token IDs")
        return value

    @field_validator("tokenizer_name")
    @classmethod
    def validate_tokenizer_name(cls, value: str) -> str:
        if not is_starcoder2_tokenizer_id(value):
            raise ValueError("tokenizer_name must reference a StarCoder2 tokenizer")
        return value

    def model_post_init(self, __context: Any) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "invalid model config: hidden_size must be divisible by num_attention_heads "
                f"(hidden_size={self.hidden_size}, num_attention_heads={self.num_attention_heads})"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "invalid model config: num_attention_heads must be divisible by "
                "num_key_value_heads for GQA "
                f"(num_attention_heads={self.num_attention_heads}, "
                f"num_key_value_heads={self.num_key_value_heads})"
            )
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("invalid model config: hidden_size must equal heads * head_dim")
        if self.declared_head_dim is not None and self.declared_head_dim != self.head_dim:
            raise ValueError(
                "invalid model config: hidden_size must equal n_query_heads * head_dim "
                f"(hidden_size={self.hidden_size}, n_query_heads={self.num_attention_heads}, "
                f"head_dim={self.declared_head_dim})"
            )
        if self.mlp_hidden_size is not None and self.mlp_hidden_size != self.intermediate_size:
            raise ValueError(
                "invalid model config: mlp_hidden_size must match intermediate_size "
                f"(mlp_hidden_size={self.mlp_hidden_size}, "
                f"intermediate_size={self.intermediate_size})"
            )
        if self.size_label in {"354m", "420m", "480m"}:
            if self.hidden_size != 768:
                raise ValueError(
                    "invalid model config: deep-thin candidate hidden_size must be 768"
                )
            if self.max_position_embeddings not in {1024, 2048}:
                raise ValueError(
                    "invalid model config: deep-thin candidate max_seq_len must be 1024 or 2048"
                )
            if self.activation != "swiglu":
                raise ValueError(
                    "invalid model config: deep-thin candidate activation must be swiglu"
                )
        if self.size_label == "480m" and self.num_hidden_layers != 64:
            raise ValueError("invalid model config: CodeWriter-Tutor-480M must use 64 layers")
        if not self.tie_word_embeddings:
            raise ValueError("invalid model config: tie_word_embeddings must be true")
        if not self.qkv_bias:
            raise ValueError("invalid model config: qkv_bias must be true")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def gqa_group_size(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    config_type: Literal["runtime"]
    platform: str
    device: str = "cpu"

    @field_validator("platform")
    @classmethod
    def validate_platform(cls, value: str) -> str:
        if value not in VALID_PLATFORMS:
            raise ValueError(f"platform must be one of {sorted(VALID_PLATFORMS)}")
        return value


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    config_type: Literal["train"]
    training_stage: str
    platform: str
    model_config_path: str
    max_seq_len: int
    drop_last: bool = True
    static_shapes: bool = True

    @field_validator("training_stage")
    @classmethod
    def validate_stage(cls, value: str) -> str:
        if value not in VALID_TRAINING_STAGES:
            raise ValueError(f"training_stage must be one of {sorted(VALID_TRAINING_STAGES)}")
        return value

    @field_validator("platform")
    @classmethod
    def validate_platform(cls, value: str) -> str:
        if value not in VALID_PLATFORMS:
            raise ValueError(f"platform must be one of {sorted(VALID_PLATFORMS)}")
        return value


class GenericConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    config_type: str = Field(default="generic")


CONFIG_MODELS: dict[str, type[BaseModel]] = {
    "model": ModelConfig,
    "runtime": RuntimeConfig,
    "train": TrainConfig,
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except OSError as exc:
        raise ConfigError(f"could not read config {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(loaded, dict):
        raise ConfigError(f"config {config_path} must contain a YAML mapping at the top level")
    return loaded


def parse_config(data: dict[str, Any]) -> BaseModel:
    config_type = data.get("config_type", "generic")
    model_cls = CONFIG_MODELS.get(str(config_type), GenericConfig)
    try:
        return model_cls.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid {config_type} config: {exc}") from exc
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def load_config(path: str | Path) -> dict[str, Any]:
    parsed = parse_config(load_yaml(path))
    return parsed.model_dump()


def load_model_config(path: str | Path) -> ModelConfig:
    parsed = parse_config(load_yaml(path))
    if not isinstance(parsed, ModelConfig):
        raise ConfigError(f"expected model config in {path}, got {type(parsed).__name__}")
    return parsed
