from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cw360.config import load_config
from cw360.data.batching import CausalLMBatch, build_causal_lm_batch
from cw360.data.dedupe import ExactDeduper
from cw360.data.extractors import extract_training_example
from cw360.data.filters import estimate_token_count
from cw360.data.fim import format_fim_code
from cw360.data.instruction import format_instruction
from cw360.data.mixture import MixtureSampler, MixtureStats
from cw360.data.modes import DataPipelineMode, parse_pipeline_mode
from cw360.data.packing import (
    PackedTokenSequence,
    continuous_pack,
    example_atomic_pack,
)
from cw360.data.quality_tiers import apply_decision
from cw360.data.schemas import ExtractedRecord, TrainingExample
from cw360.data.source_filters import filter_training_example
from cw360.data.sources import DatasetSourceConfig, load_dataset_source_configs
from cw360.data.stage_mixtures import stage_weights
from cw360.data.streaming import iter_source_records
from cw360.data.tokenize import TokenizedExample, TokenizerLike, tokenize_example
from cw360.utils.hashing import sha256_bytes

RawSourceItem = TrainingExample | ExtractedRecord | Mapping[str, Any]

PYTHON_SOURCES = frozenset({"stack_v2_python", "codesearchnet_python", "debugbench", "synthetic"})


@dataclass(frozen=True, slots=True)
class PipelineOptions:
    mode: DataPipelineMode = DataPipelineMode.TINY_LOCAL_SYNTHETIC
    training_stage: str = "base_pretrain"
    seed: int = 0
    max_seq_len: int = 512
    batch_size: int = 2
    packing_mode: str = "continuous"
    allow_exhausted_sources: bool = True
    fim_probability: float = 1.0
    tier_sampling_rates: Mapping[str, float] = field(
        default_factory=lambda: {"A": 1.0, "B": 1.0, "RESCUE": 0.0, "C": 0.0}
    )
    tier_loss_multipliers: Mapping[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class DataPipeline:
    source_iterators: Mapping[str, Iterable[RawSourceItem]]
    options: PipelineOptions
    tokenizer: TokenizerLike | None = None
    weights: Mapping[str, float] | None = None
    _last_sampler: MixtureSampler[TrainingExample] | None = field(
        default=None, init=False, repr=False
    )

    def candidate_stream(self) -> Iterator[TrainingExample]:
        if parse_pipeline_mode(self.options.mode) is not DataPipelineMode.CANDIDATE_STREAM:
            # Candidate streams are also useful in tiny smoke tests, but keep the method explicit.
            pass
        yield from self._sample_examples(format_for_prepack=False)

    def prepack_stream(self) -> Iterator[TrainingExample]:
        yield from self._sample_examples(format_for_prepack=True)

    def eval_stream(self) -> Iterator[TrainingExample]:
        yield from self._sample_examples(format_for_prepack=True)

    def examples_for_mode(self) -> Iterator[TrainingExample]:
        if parse_pipeline_mode(self.options.mode) is DataPipelineMode.CANDIDATE_STREAM:
            yield from self.candidate_stream()
            return
        yield from self.prepack_stream()

    def tokenized_stream(self, *, preserve_text: bool = False) -> Iterator[TokenizedExample]:
        if self.tokenizer is None:
            raise ValueError("tokenized_stream requires a tokenizer")
        for example in self.prepack_stream():
            yield tokenize_example(example, self.tokenizer, preserve_text=preserve_text)

    def packed_stream(self) -> Iterator[PackedTokenSequence]:
        if self.tokenizer is None:
            raise ValueError("packed_stream requires a tokenizer")
        packer = (
            continuous_pack
            if self.options.packing_mode == "continuous"
            else example_atomic_pack
        )
        tokenized = self.tokenized_stream()
        while True:
            chunk = _take(tokenized, self.options.batch_size * 8)
            if not chunk:
                return
            yield from packer(chunk, max_seq_len=self.options.max_seq_len, tpu_prepack=False)

    def batch_stream(self, *, pad_token_id: int) -> Iterator[CausalLMBatch]:
        packed = self.packed_stream()
        while True:
            sequences = _take(packed, self.options.batch_size)
            if not sequences:
                return
            if len(sequences) < self.options.batch_size:
                return
            yield build_causal_lm_batch(
                sequences,
                seq_len=self.options.max_seq_len,
                pad_token_id=pad_token_id,
            )

    def mixture_stats(self) -> MixtureStats | None:
        if self._last_sampler is None:
            return None
        return self._last_sampler.stats()

    def _sample_examples(self, *, format_for_prepack: bool) -> Iterator[TrainingExample]:
        source_examples = {
            name: self._processed_source(name, rows, format_for_prepack=format_for_prepack)
            for name, rows in self.source_iterators.items()
        }
        sampler = MixtureSampler(
            source_examples,
            self._active_weights(),
            seed=self.options.seed,
            allow_exhausted=self.options.allow_exhausted_sources,
            tier_sampling_rates=self.options.tier_sampling_rates,
            tier_loss_multipliers=self.options.tier_loss_multipliers,
        )
        self._last_sampler = sampler
        yield from sampler

    def _active_weights(self) -> dict[str, float]:
        configured = dict(self.weights or stage_weights(self.options.training_stage))
        return {
            name: weight
            for name, weight in configured.items()
            if name in self.source_iterators and weight > 0
        }

    def _processed_source(
        self,
        source_name: str,
        rows: Iterable[RawSourceItem],
        *,
        format_for_prepack: bool,
    ) -> Iterator[TrainingExample]:
        deduper = ExactDeduper()
        for item in rows:
            example = _coerce_training_example(source_name, item)
            if example is None:
                continue
            decision = filter_training_example(example, deduper=deduper)
            if not decision.accepted:
                continue
            apply_decision(example, decision)
            example.training_stage = _training_stage_for_pipeline(
                example,
                fallback=self.options.training_stage,
            )
            _enrich_metadata(example, tokenizer=self.tokenizer)
            if format_for_prepack and not example.metadata.get("preformatted_text"):
                example = _format_for_stage(
                    example,
                    training_stage=example.training_stage,
                    tokenizer=self.tokenizer,
                    fim_probability=self.options.fim_probability,
                )
                _enrich_metadata(example, tokenizer=self.tokenizer)
            yield example


def build_pipeline_from_config(
    config_path: str | Path,
    source_iterators: Mapping[str, Iterable[RawSourceItem]],
    *,
    mode: str | DataPipelineMode,
    tokenizer: TokenizerLike | None = None,
    weights: Mapping[str, float] | None = None,
    seed: int = 0,
) -> DataPipeline:
    config = load_config(config_path)
    options = PipelineOptions(
        mode=parse_pipeline_mode(mode),
        training_stage=str(config.get("training_stage", "base_pretrain")),
        seed=seed,
        max_seq_len=int(config.get("max_seq_len", 512)),
        batch_size=int(config.get("micro_batch_size", 2)),
    )
    return DataPipeline(source_iterators, options, tokenizer=tokenizer, weights=weights)


def source_iterators_from_dataset_configs(
    configs: Mapping[str, DatasetSourceConfig],
    *,
    source_names: Iterable[str] | None = None,
) -> dict[str, Iterable[ExtractedRecord]]:
    selected = set(source_names) if source_names is not None else set(configs)
    missing = selected.difference(configs)
    if missing:
        raise ValueError(f"unknown dataset source config(s): {sorted(missing)}")
    return {name: iter_source_records(configs[name]) for name in selected}


def source_iterators_from_dataset_config_path(
    config_path: str | Path,
    *,
    source_names: Iterable[str] | None = None,
) -> dict[str, Iterable[ExtractedRecord]]:
    path = Path(config_path)
    configs = {
        name: _resolve_local_source_path(config, path)
        for name, config in load_dataset_source_configs(path).items()
    }
    return source_iterators_from_dataset_configs(configs, source_names=source_names)


def _coerce_training_example(source_name: str, item: RawSourceItem) -> TrainingExample | None:
    if isinstance(item, ExtractedRecord):
        return item.example
    if isinstance(item, TrainingExample):
        return item
    if isinstance(item, Mapping):
        example = extract_training_example(source_name, item)
        example.metadata.setdefault("original_fields", _safe_original_fields(item))
        return example
    raise TypeError(f"unsupported source item type for {source_name}: {type(item).__name__}")


def _resolve_local_source_path(
    config: DatasetSourceConfig,
    config_path: Path,
) -> DatasetSourceConfig:
    if not config.local_path:
        return config
    local_path = Path(config.local_path)
    if not local_path.is_absolute():
        local_path = config_path.resolve().parent.parent / local_path
    return DatasetSourceConfig(
        name=config.name,
        dataset_id=config.dataset_id,
        subset=config.subset,
        split=config.split,
        streaming=config.streaming,
        extractor=config.extractor,
        local_path=str(local_path),
        gated=config.gated,
        max_preview_examples=config.max_preview_examples,
        extra=config.extra,
    )


def _safe_original_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    raw_text_keys = {
        "content",
        "text",
        "code",
        "func_code",
        "function",
        "prompt",
        "completion",
        "messages",
        "buggy_code",
        "fixed_code",
        "stack_trace",
        "traceback",
    }
    for key, value in row.items():
        key_str = str(key)
        if key_str in raw_text_keys:
            continue
        if isinstance(value, str):
            if len(value) <= 500 and key_str.lower() not in {"token", "api_key", "secret"}:
                safe[key_str] = value
        elif isinstance(value, int | float | bool) or value is None:
            safe[key_str] = value
    return safe


def _enrich_metadata(example: TrainingExample, *, tokenizer: TokenizerLike | None) -> None:
    metadata = example.metadata
    metadata["source"] = example.source
    metadata["example_type"] = example.example_type
    metadata.setdefault("language", "python" if example.source in PYTHON_SOURCES else "text")
    metadata["quality_tier"] = example.quality_tier
    metadata["estimated_length_chars"] = len(example.text)
    if tokenizer is not None:
        metadata["estimated_tokens"] = len(tokenizer.encode(example.text, add_special_tokens=False))
    else:
        metadata["estimated_tokens"] = estimate_token_count(example.text)
    metadata["training_stage"] = example.training_stage
    metadata["loss_weight"] = example.loss_weight
    metadata.setdefault("stable_input_hash", _stable_input_hash(example))


def _stable_input_hash(example: TrainingExample) -> str:
    payload = "\0".join([example.source, example.example_type, example.text])
    return sha256_bytes(payload.encode("utf-8"))


def _training_stage_for_pipeline(example: TrainingExample, *, fallback: str) -> str:
    if example.metadata.get("prepack_ready"):
        stage = example.metadata.get("train_stage") or example.training_stage
        if isinstance(stage, str) and stage:
            return stage
    return fallback


def _format_for_stage(
    example: TrainingExample,
    *,
    training_stage: str,
    tokenizer: TokenizerLike | None,
    fim_probability: float,
) -> TrainingExample:
    if training_stage == "fim_train" and example.example_type in {
        "python_code",
        "docstring_code_pair",
    }:
        formatted = format_fim_code(
            example.text,
            tokenizer=tokenizer,
            fim_probability=fim_probability,
        )
        example.text = formatted.text
        example.metadata["fim_applied"] = formatted.applied
        example.metadata["fim_tokenizer_supported"] = formatted.markers.tokenizer_supported
        return example

    if training_stage == "instruction_tune":
        prompt = example.metadata.get("prompt")
        response = example.metadata.get("response") or example.metadata.get("completion")
        task_type = str(example.metadata.get("task_type") or "code_writing")
        if isinstance(prompt, str) and isinstance(response, str):
            formatted = format_instruction(prompt, response, task_type=task_type)
            example.text = formatted.text
            example.example_type = f"instruction_{formatted.task_type}"
            example.metadata["formatted_instruction"] = True
    return example


def _take(iterator: Iterator[Any], count: int) -> list[Any]:
    items: list[Any] = []
    for _ in range(count):
        try:
            items.append(next(iterator))
        except StopIteration:
            break
    return items
