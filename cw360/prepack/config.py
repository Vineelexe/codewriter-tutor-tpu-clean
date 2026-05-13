from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from cw360.tokenizer.special_tokens import DEFAULT_TOKENIZER_ID, is_starcoder2_tokenizer_id


class PrepackConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PrepackSection:
    dataset_name: str
    model_size_target: str
    max_seq_len: int
    seq_len_plus_one: int
    token_dtype: str
    shard_format: str
    shard_num_sequences: int
    rolling_shuffle_buffer_sequences: int
    rolling_shuffle_buffer_memory_guard: bool
    rolling_shuffle_buffer_max_ram_fraction: float
    minimum_shuffle_buffer_sequences: int
    output_dir: str
    seed: int
    drop_incomplete_sequence: bool
    verify_uint16: bool
    write_per_shard_metadata: bool
    write_global_manifest: bool
    write_kaggle_dataset_metadata: bool
    tokenizer_id: str = DEFAULT_TOKENIZER_ID
    dataset_version: str = "v1"
    kaggle_owner: str = "vineel"


@dataclass(frozen=True, slots=True)
class SplitsConfig:
    train_fraction: float
    val_fraction: float
    test_fraction: float
    split_by: str
    synthetic_split_key: str


@dataclass(frozen=True, slots=True)
class QualityConfig:
    tier_a_weight: float
    tier_b_weight: float
    rescue_policy: str

    @property
    def tier_weights(self) -> dict[str, float]:
        return {
            "A": self.tier_a_weight,
            "B": self.tier_b_weight,
            "TIER_A": self.tier_a_weight,
            "TIER_B": self.tier_b_weight,
        }


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    shard_mixture_tolerance_abs: float
    fail_on_token_overflow: bool
    fail_on_empty_shard: bool
    fail_on_missing_manifest: bool
    fail_if_split_fractions_do_not_sum_to_one: bool
    fail_if_atomic_examples_cross_sequence_boundary: bool


@dataclass(frozen=True, slots=True)
class PrepackFactoryConfig:
    prepack: PrepackSection
    mixture: dict[str, float]
    splits: SplitsConfig
    quality: QualityConfig
    validation: ValidationConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_overrides(
        self,
        *,
        output_dir: str | Path | None = None,
        shard_num_sequences: int | None = None,
        rolling_shuffle_buffer_sequences: int | None = None,
        minimum_shuffle_buffer_sequences: int | None = None,
    ) -> PrepackFactoryConfig:
        prepack = self.prepack
        if output_dir is not None:
            prepack = replace(prepack, output_dir=str(output_dir))
        if shard_num_sequences is not None:
            prepack = replace(prepack, shard_num_sequences=int(shard_num_sequences))
        if rolling_shuffle_buffer_sequences is not None:
            prepack = replace(
                prepack,
                rolling_shuffle_buffer_sequences=int(rolling_shuffle_buffer_sequences),
            )
        if minimum_shuffle_buffer_sequences is not None:
            prepack = replace(
                prepack,
                minimum_shuffle_buffer_sequences=int(minimum_shuffle_buffer_sequences),
            )
        return replace(self, prepack=prepack)


def load_prepack_config(path: str | Path) -> PrepackFactoryConfig:
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PrepackConfigError(f"could not read prepack config {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise PrepackConfigError(f"invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise PrepackConfigError("prepack config must be a mapping")

    config = parse_prepack_config(raw)
    validate_prepack_config(config)
    return config


def parse_prepack_config(raw: dict[str, Any]) -> PrepackFactoryConfig:
    prepack_raw = _mapping(raw, "prepack")
    mixture_raw = _mapping(raw, "mixture")
    splits_raw = _mapping(raw, "splits")
    quality_raw = _mapping(raw, "quality")
    validation_raw = _mapping(raw, "validation")

    prepack = PrepackSection(
        dataset_name=_str(prepack_raw, "dataset_name"),
        model_size_target=_str(prepack_raw, "model_size_target"),
        max_seq_len=_int(prepack_raw, "max_seq_len"),
        seq_len_plus_one=_int(prepack_raw, "seq_len_plus_one"),
        token_dtype=_str(prepack_raw, "token_dtype"),
        shard_format=_str(prepack_raw, "shard_format"),
        shard_num_sequences=_int(prepack_raw, "shard_num_sequences"),
        rolling_shuffle_buffer_sequences=_int(
            prepack_raw,
            "rolling_shuffle_buffer_sequences",
        ),
        rolling_shuffle_buffer_memory_guard=_bool(
            prepack_raw,
            "rolling_shuffle_buffer_memory_guard",
        ),
        rolling_shuffle_buffer_max_ram_fraction=_float(
            prepack_raw,
            "rolling_shuffle_buffer_max_ram_fraction",
        ),
        minimum_shuffle_buffer_sequences=_int(
            prepack_raw,
            "minimum_shuffle_buffer_sequences",
        ),
        output_dir=_str(prepack_raw, "output_dir"),
        seed=_int(prepack_raw, "seed"),
        drop_incomplete_sequence=_bool(prepack_raw, "drop_incomplete_sequence"),
        verify_uint16=_bool(prepack_raw, "verify_uint16"),
        write_per_shard_metadata=_bool(prepack_raw, "write_per_shard_metadata"),
        write_global_manifest=_bool(prepack_raw, "write_global_manifest"),
        write_kaggle_dataset_metadata=_bool(
            prepack_raw,
            "write_kaggle_dataset_metadata",
        ),
        tokenizer_id=str(prepack_raw.get("tokenizer_id", DEFAULT_TOKENIZER_ID)),
        dataset_version=str(prepack_raw.get("dataset_version", "v1")),
        kaggle_owner=str(prepack_raw.get("kaggle_owner", "vineel")),
    )
    mixture = {str(name): float(weight) for name, weight in mixture_raw.items()}
    splits = SplitsConfig(
        train_fraction=_float(splits_raw, "train_fraction"),
        val_fraction=_float(splits_raw, "val_fraction"),
        test_fraction=_float(splits_raw, "test_fraction"),
        split_by=_str(splits_raw, "split_by"),
        synthetic_split_key=_str(splits_raw, "synthetic_split_key"),
    )
    quality = QualityConfig(
        tier_a_weight=_float(quality_raw, "tier_a_weight"),
        tier_b_weight=_float(quality_raw, "tier_b_weight"),
        rescue_policy=_str(quality_raw, "rescue_policy"),
    )
    validation = ValidationConfig(
        shard_mixture_tolerance_abs=_float(
            validation_raw,
            "shard_mixture_tolerance_abs",
        ),
        fail_on_token_overflow=_bool(validation_raw, "fail_on_token_overflow"),
        fail_on_empty_shard=_bool(validation_raw, "fail_on_empty_shard"),
        fail_on_missing_manifest=_bool(validation_raw, "fail_on_missing_manifest"),
        fail_if_split_fractions_do_not_sum_to_one=_bool(
            validation_raw,
            "fail_if_split_fractions_do_not_sum_to_one",
        ),
        fail_if_atomic_examples_cross_sequence_boundary=_bool(
            validation_raw,
            "fail_if_atomic_examples_cross_sequence_boundary",
        ),
    )
    return PrepackFactoryConfig(
        prepack=prepack,
        mixture=mixture,
        splits=splits,
        quality=quality,
        validation=validation,
    )


def validate_prepack_config(config: PrepackFactoryConfig) -> None:
    prepack = config.prepack
    if prepack.max_seq_len <= 0:
        raise PrepackConfigError("max_seq_len must be positive")
    if prepack.seq_len_plus_one != prepack.max_seq_len + 1:
        raise PrepackConfigError("seq_len_plus_one must equal max_seq_len + 1")
    if not is_starcoder2_tokenizer_id(prepack.tokenizer_id):
        raise PrepackConfigError("tokenizer_id must reference a StarCoder2 tokenizer")
    if prepack.token_dtype != "uint16":
        raise PrepackConfigError("token_dtype must be uint16 for TPU prepacked shards")
    if prepack.shard_format != "npy":
        raise PrepackConfigError("shard_format must be npy")
    if prepack.shard_num_sequences <= 0:
        raise PrepackConfigError("shard_num_sequences must be positive")
    if prepack.rolling_shuffle_buffer_sequences <= 0:
        raise PrepackConfigError("rolling_shuffle_buffer_sequences must be positive")
    if not 0.0 < prepack.rolling_shuffle_buffer_max_ram_fraction <= 1.0:
        raise PrepackConfigError("rolling_shuffle_buffer_max_ram_fraction must be in (0, 1]")
    mixture_total = sum(config.mixture.values())
    if not config.mixture or mixture_total <= 0:
        raise PrepackConfigError("mixture must contain at least one positive weight")
    if any(weight < 0 for weight in config.mixture.values()):
        raise PrepackConfigError("mixture weights must be non-negative")
    if abs(mixture_total - 1.0) > 1.0e-9:
        raise PrepackConfigError("mixture weights must sum to 1.0")
    split_total = (
        config.splits.train_fraction
        + config.splits.val_fraction
        + config.splits.test_fraction
    )
    if abs(split_total - 1.0) > 1.0e-9:
        raise PrepackConfigError("split fractions must sum to 1.0")


def _mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise PrepackConfigError(f"prepack config must contain {key!r} mapping")
    return value


def _str(raw: dict[str, Any], key: str) -> str:
    if key not in raw:
        raise PrepackConfigError(f"missing required prepack key {key!r}")
    value = raw[key]
    if not isinstance(value, str) or not value:
        raise PrepackConfigError(f"{key} must be a non-empty string")
    return value


def _int(raw: dict[str, Any], key: str) -> int:
    if key not in raw:
        raise PrepackConfigError(f"missing required prepack key {key!r}")
    return int(raw[key])


def _float(raw: dict[str, Any], key: str) -> float:
    if key not in raw:
        raise PrepackConfigError(f"missing required prepack key {key!r}")
    return float(raw[key])


def _bool(raw: dict[str, Any], key: str) -> bool:
    if key not in raw:
        raise PrepackConfigError(f"missing required prepack key {key!r}")
    return bool(raw[key])
