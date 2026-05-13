from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cw360.data.schemas import TrainingExample
from cw360.data.tokenize import TokenizedExample, TokenizerLike, tokenize_example
from cw360.prepack.config import PrepackFactoryConfig
from cw360.prepack.kaggle_dataset import write_kaggle_dataset_metadata
from cw360.prepack.manifest import (
    current_git_commit,
    hash_payload,
    merge_count_dicts,
    normalize_counts,
    sha256_file,
    utc_now_iso,
    write_json,
)
from cw360.prepack.rolling_shuffle import RollingShuffleBuffer, resolve_shuffle_buffer_sequences
from cw360.prepack.shard_index import SPLITS, SplitName, assign_split_for_key

LOSS_WEIGHT_HANDLING_POLICY = (
    "loss_weight is applied during prepack sampling/oversampling only; final .npy shards "
    "store uint16 token IDs only and TPU training uses uniform causal LM loss"
)
UINT16_MAX = 65_535


@dataclass(frozen=True, slots=True)
class _Attribution:
    source: str
    task_type: str
    training_stage: str
    quality_tier: str
    split_key: str


@dataclass(frozen=True, slots=True)
class PrepackedSequence:
    token_ids: Sequence[int] | np.ndarray
    split_key: str
    source_counts: Mapping[str, int]
    task_type_counts: Mapping[str, int] = field(default_factory=dict)
    training_stage_counts: Mapping[str, int] = field(default_factory=dict)
    quality_tier_counts: Mapping[str, int] = field(default_factory=dict)
    packing_mode_counts: Mapping[str, int] = field(default_factory=dict)
    source_example_counts: Mapping[str, int] = field(default_factory=dict)
    atomic_boundary_violations: int = 0


def sample_examples_by_loss_weight(
    examples: Iterable[TrainingExample],
    *,
    tier_weights: Mapping[str, float],
    seed: int,
) -> Iterator[TrainingExample]:
    rng = np.random.default_rng(seed)
    for index, example in enumerate(examples):
        multiplier = effective_sampling_multiplier(example, tier_weights=tier_weights)
        if multiplier <= 0:
            continue
        whole = int(multiplier)
        fractional = multiplier - whole
        copies = whole + (1 if fractional > 0 and rng.random() < fractional else 0)
        for copy_index in range(copies):
            metadata = dict(example.metadata)
            metadata["prepack_sampling_multiplier"] = multiplier
            metadata["prepack_copy_index"] = copy_index
            metadata["prepack_source_index"] = index
            yield TrainingExample(
                text=example.text,
                source=example.source,
                example_type=example.example_type,
                metadata=metadata,
                quality_tier=example.quality_tier,
                loss_weight=example.loss_weight,
                training_stage=example.training_stage,
            )


def effective_sampling_multiplier(
    example: TrainingExample,
    *,
    tier_weights: Mapping[str, float],
) -> float:
    if example.loss_weight is not None:
        return float(example.loss_weight)
    tier = str(example.quality_tier or "").upper()
    return float(tier_weights.get(tier, 1.0))


def iter_prepacked_sequences(
    tokenized_examples: Iterable[TokenizedExample],
    *,
    max_seq_len: int,
    drop_incomplete_sequence: bool = True,
) -> Iterator[PrepackedSequence]:
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    window_size = max_seq_len + 1
    continuous_tokens: list[int] = []
    continuous_attrs: list[_Attribution] = []
    atomic_tokens: list[int] = []
    atomic_attrs: list[_Attribution] = []

    for example in tokenized_examples:
        attr = _attribution(example)
        token_ids = [int(token_id) for token_id in example.token_ids]
        if _requires_atomic_pack(example):
            if len(token_ids) >= window_size:
                if atomic_tokens and not drop_incomplete_sequence:
                    yield _build_sequence(
                        atomic_tokens,
                        atomic_attrs,
                        packing_mode="example_atomic_pack",
                        atomic_boundary_violations=0,
                    )
                atomic_tokens = []
                atomic_attrs = []
                yield _build_sequence(
                    token_ids[:window_size],
                    [attr] * window_size,
                    packing_mode="example_atomic_pack",
                    atomic_boundary_violations=0,
                )
                continue

            if len(atomic_tokens) + len(token_ids) > window_size:
                atomic_tokens = []
                atomic_attrs = []
            atomic_tokens.extend(token_ids)
            atomic_attrs.extend([attr] * len(token_ids))
            if len(atomic_tokens) == window_size:
                yield _build_sequence(
                    atomic_tokens,
                    atomic_attrs,
                    packing_mode="example_atomic_pack",
                    atomic_boundary_violations=0,
                )
                atomic_tokens = []
                atomic_attrs = []
            continue

        continuous_tokens.extend(token_ids)
        continuous_attrs.extend([attr] * len(token_ids))
        while len(continuous_tokens) >= window_size:
            window_tokens = continuous_tokens[:window_size]
            window_attrs = continuous_attrs[:window_size]
            yield _build_sequence(
                window_tokens,
                window_attrs,
                packing_mode="continuous_pack",
                atomic_boundary_violations=0,
            )
            continuous_tokens = continuous_tokens[window_size:]
            continuous_attrs = continuous_attrs[window_size:]

    if not drop_incomplete_sequence:
        if continuous_tokens:
            yield _build_sequence(
                continuous_tokens,
                continuous_attrs,
                packing_mode="continuous_pack",
                atomic_boundary_violations=0,
            )
        if atomic_tokens:
            yield _build_sequence(
                atomic_tokens,
                atomic_attrs,
                packing_mode="example_atomic_pack",
                atomic_boundary_violations=0,
            )


def tokenize_and_pack_examples(
    examples: Iterable[TrainingExample],
    tokenizer: TokenizerLike,
    *,
    config: PrepackFactoryConfig,
) -> Iterator[PrepackedSequence]:
    weighted_examples = sample_examples_by_loss_weight(
        examples,
        tier_weights=config.quality.tier_weights,
        seed=config.prepack.seed,
    )
    tokenized = (
        tokenize_example(example, tokenizer, preserve_text=False)
        for example in weighted_examples
    )
    yield from iter_prepacked_sequences(
        tokenized,
        max_seq_len=config.prepack.max_seq_len,
        drop_incomplete_sequence=config.prepack.drop_incomplete_sequence,
    )


@dataclass(slots=True)
class _SplitShardWriter:
    split: SplitName
    output_dir: Path
    config: PrepackFactoryConfig
    config_hash: str
    tokenizer_id: str
    tokenizer_hash: str | None
    data_snapshot_hash: str
    effective_sampling_multipliers: Mapping[str, float]
    _buffer: list[PrepackedSequence] = field(default_factory=list, init=False)
    _shard_index: int = field(default=0, init=False)
    shard_metadata: list[dict[str, Any]] = field(default_factory=list, init=False)

    def add(self, sequence: PrepackedSequence) -> None:
        self._validate_sequence(sequence)
        self._buffer.append(sequence)
        if len(self._buffer) >= self.config.prepack.shard_num_sequences:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        split_dir = self.output_dir / self.split
        split_dir.mkdir(parents=True, exist_ok=True)
        filename = f"chunk_{self._shard_index:06d}.npy"
        path = split_dir / filename
        array = np.asarray([sequence.token_ids for sequence in self._buffer], dtype=np.uint16)
        np.save(path, array, allow_pickle=False)
        digest = sha256_file(path)
        relative = f"{self.split}/{filename}"
        metadata = self._build_metadata(
            relative_filename=relative,
            array=array,
            shard_hash=digest,
        )
        if self.config.prepack.write_per_shard_metadata:
            write_json(path.with_suffix(".json"), metadata)
        self.shard_metadata.append(metadata)
        self._buffer = []
        self._shard_index += 1

    def _validate_sequence(self, sequence: PrepackedSequence) -> None:
        if len(sequence.token_ids) != self.config.prepack.seq_len_plus_one:
            raise ValueError(
                "prepacked TPU sequence length must equal seq_len_plus_one "
                f"({self.config.prepack.seq_len_plus_one})"
            )
        if not self.config.prepack.verify_uint16:
            return
        for token_id in sequence.token_ids:
            value = int(token_id)
            if value < 0 or value > UINT16_MAX:
                raise ValueError(f"token id {value} cannot be stored as uint16")

    def _build_metadata(
        self,
        *,
        relative_filename: str,
        array: np.ndarray,
        shard_hash: str,
    ) -> dict[str, Any]:
        source_counts = _merge_sequence_counts(self._buffer, "source_counts")
        task_type_counts = _merge_sequence_counts(self._buffer, "task_type_counts")
        training_stage_counts = _merge_sequence_counts(self._buffer, "training_stage_counts")
        quality_tier_counts = _merge_sequence_counts(self._buffer, "quality_tier_counts")
        packing_mode_counts = _merge_sequence_counts(self._buffer, "packing_mode_counts")
        source_example_counts = _merge_sequence_counts(self._buffer, "source_example_counts")
        return {
            "shard_filename": relative_filename,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "seq_len_plus_one": self.config.prepack.seq_len_plus_one,
            "num_sequences": int(array.shape[0]),
            "token_count": int(array.shape[0] * array.shape[1]),
            "split": self.split,
            "source_mixture_actual": normalize_counts(source_counts),
            "task_type_mixture_actual": normalize_counts(task_type_counts),
            "train_stage_mixture_actual": normalize_counts(training_stage_counts),
            "quality_tier_counts": quality_tier_counts,
            "packing_mode_counts": packing_mode_counts,
            "loss_weight_handling_summary": LOSS_WEIGHT_HANDLING_POLICY,
            "effective_sampling_multipliers": dict(self.effective_sampling_multipliers),
            "source_example_counts": source_example_counts,
            "source_counts": source_counts,
            "task_type_counts": task_type_counts,
            "training_stage_counts": training_stage_counts,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_hash": self.tokenizer_hash,
            "config_hash": self.config_hash,
            "data_snapshot_hash": self.data_snapshot_hash,
            "sha256": shard_hash,
            "atomic_boundary_violations": sum(
                sequence.atomic_boundary_violations for sequence in self._buffer
            ),
            "created_at": utc_now_iso(),
        }


@dataclass(slots=True)
class PrepackDatasetWriter:
    config: PrepackFactoryConfig
    output_dir: str | Path | None = None
    tokenizer_hash: str | None = None
    data_snapshot_hash: str = "unknown"
    source_dataset_versions: Mapping[str, str] | None = None
    synthetic_repo_id: str | None = None
    synthetic_manifest_id: str | None = None
    split_manifest_id: str | None = None
    code_git_commit: str | None = None
    _output_dir: Path = field(init=False)
    _config_hash: str = field(init=False)
    _split_writers: dict[SplitName, _SplitShardWriter] = field(init=False)
    _shufflers: dict[SplitName, RollingShuffleBuffer[PrepackedSequence]] = field(init=False)
    _closed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._output_dir = Path(self.output_dir or self.config.prepack.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            (self._output_dir / split).mkdir(parents=True, exist_ok=True)
        self._config_hash = hash_payload(self.config.to_dict())
        multipliers = {
            "tier_a_weight": self.config.quality.tier_a_weight,
            "tier_b_weight": self.config.quality.tier_b_weight,
            "loss_weight_policy": "example_loss_weight_overrides_tier_weight",
        }
        self._split_writers = {
            split: _SplitShardWriter(
                split=split,
                output_dir=self._output_dir,
                config=self.config,
                config_hash=self._config_hash,
                tokenizer_id=self.config.prepack.tokenizer_id,
                tokenizer_hash=self.tokenizer_hash,
                data_snapshot_hash=self.data_snapshot_hash,
                effective_sampling_multipliers=multipliers,
            )
            for split in SPLITS
        }
        self._shufflers = {}
        for index, split in enumerate(SPLITS):
            buffer_size = resolve_shuffle_buffer_sequences(
                configured_sequences=self.config.prepack.rolling_shuffle_buffer_sequences,
                minimum_sequences=self.config.prepack.minimum_shuffle_buffer_sequences,
                seq_len_plus_one=self.config.prepack.seq_len_plus_one,
                max_ram_fraction=self.config.prepack.rolling_shuffle_buffer_max_ram_fraction,
                memory_guard=self.config.prepack.rolling_shuffle_buffer_memory_guard,
            )
            self._shufflers[split] = RollingShuffleBuffer(
                buffer_size=buffer_size,
                seed=self.config.prepack.seed + index,
                emit=self._split_writers[split].add,
            )

    def add_sequence(self, sequence: PrepackedSequence, *, split: SplitName | None = None) -> None:
        if self._closed:
            raise RuntimeError("cannot add sequence after writer is closed")
        target_split = split or assign_split_for_key(
            sequence.split_key,
            train_fraction=self.config.splits.train_fraction,
            val_fraction=self.config.splits.val_fraction,
            test_fraction=self.config.splits.test_fraction,
        )
        self._split_writers[target_split]._validate_sequence(sequence)
        self._shufflers[target_split].add(sequence)

    def add_sequences(
        self,
        sequences: Iterable[PrepackedSequence],
        *,
        target_sequences: int | None = None,
    ) -> int:
        written = 0
        for sequence in sequences:
            if target_sequences is not None and written >= target_sequences:
                break
            self.add_sequence(sequence)
            written += 1
        return written

    def close(self) -> dict[str, Any]:
        if self._closed:
            return self._load_manifest()
        for split in SPLITS:
            self._shufflers[split].close()
            self._split_writers[split].flush()
        manifest = self._build_global_manifest()
        if self.config.prepack.write_global_manifest:
            write_json(self._output_dir / "manifest.json", manifest)
        if self.config.prepack.write_kaggle_dataset_metadata:
            write_kaggle_dataset_metadata(self._output_dir, self.config)
        self._closed = True
        return manifest

    def _load_manifest(self) -> dict[str, Any]:
        from cw360.prepack.manifest import read_json

        return read_json(self._output_dir / "manifest.json")

    def _build_global_manifest(self) -> dict[str, Any]:
        shard_rows = self._all_shard_metadata()
        shard_file_list: dict[str, list[str]] = {split: [] for split in SPLITS}
        shard_hashes: dict[str, dict[str, str]] = {split: {} for split in SPLITS}
        for row in shard_rows:
            split = str(row["split"])
            filename = str(row["shard_filename"])
            shard_file_list[split].append(filename)
            shard_hashes[split][filename] = str(row["sha256"])

        split_counts = {
            split: sum(
                int(row["num_sequences"])
                for row in self._split_writers[split].shard_metadata
            )
            for split in SPLITS
        }
        total_sequences = sum(split_counts.values())
        source_counts = merge_count_dicts([row.get("source_counts", {}) for row in shard_rows])
        per_shard_summary = {
            row["shard_filename"]: row["source_mixture_actual"] for row in shard_rows
        }
        manifest = {
            "dataset_name": self.config.prepack.dataset_name,
            "dataset_version": self.config.prepack.dataset_version,
            "created_at": utc_now_iso(),
            "tokenizer_id": self.config.prepack.tokenizer_id,
            "tokenizer_config_hash": self.tokenizer_hash,
            "max_seq_len": self.config.prepack.max_seq_len,
            "seq_len_plus_one": self.config.prepack.seq_len_plus_one,
            "dtype": self.config.prepack.token_dtype,
            "shard_format": self.config.prepack.shard_format,
            "shard_count": len(shard_rows),
            "total_sequences": total_sequences,
            "total_tokens_for_training": total_sequences * self.config.prepack.max_seq_len,
            "total_token_ids_stored": total_sequences * self.config.prepack.seq_len_plus_one,
            "shard_file_list": shard_file_list,
            "shard_hashes": shard_hashes,
            "split_fractions": {
                "train": self.config.splits.train_fraction,
                "val": self.config.splits.val_fraction,
                "test": self.config.splits.test_fraction,
            },
            "actual_split_counts": split_counts,
            "actual_split_fractions": normalize_counts(split_counts),
            "target_mixture": dict(self.config.mixture),
            "actual_global_mixture": normalize_counts(source_counts),
            "per_shard_mixture_summary": per_shard_summary,
            "validation": {
                "shard_mixture_tolerance_abs": (
                    self.config.validation.shard_mixture_tolerance_abs
                ),
                "fail_on_token_overflow": self.config.validation.fail_on_token_overflow,
                "fail_on_empty_shard": self.config.validation.fail_on_empty_shard,
                "fail_on_missing_manifest": self.config.validation.fail_on_missing_manifest,
                "fail_if_split_fractions_do_not_sum_to_one": (
                    self.config.validation.fail_if_split_fractions_do_not_sum_to_one
                ),
                "fail_if_atomic_examples_cross_sequence_boundary": (
                    self.config.validation.fail_if_atomic_examples_cross_sequence_boundary
                ),
            },
            "loss_weight_handling_policy": LOSS_WEIGHT_HANDLING_POLICY,
            "effective_sampling_multipliers": {
                "tier_a_weight": self.config.quality.tier_a_weight,
                "tier_b_weight": self.config.quality.tier_b_weight,
                "loss_weight_policy": "example_loss_weight_overrides_tier_weight",
            },
            "source_dataset_versions": dict(self.source_dataset_versions or {}),
            "synthetic_repo_id": self.synthetic_repo_id,
            "synthetic_manifest_id": self.synthetic_manifest_id,
            "split_manifest_id": self.split_manifest_id,
            "prepack_config_hash": self._config_hash,
            "data_snapshot_hash": self.data_snapshot_hash,
            "code_git_commit": self.code_git_commit or current_git_commit(Path.cwd()),
            "shards": shard_rows,
        }
        manifest["manifest_hash"] = hash_payload(
            {key: value for key, value in manifest.items() if key != "manifest_hash"}
        )
        return manifest

    def _all_shard_metadata(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for split in SPLITS:
            rows.extend(self._split_writers[split].shard_metadata)
        return rows


def write_prepacked_dataset_from_examples(
    examples: Iterable[TrainingExample],
    tokenizer: TokenizerLike,
    *,
    config: PrepackFactoryConfig,
    output_dir: str | Path | None = None,
    target_sequences: int | None = None,
    tokenizer_hash: str | None = None,
    data_snapshot_hash: str = "unknown",
    source_dataset_versions: Mapping[str, str] | None = None,
    synthetic_repo_id: str | None = None,
    synthetic_manifest_id: str | None = None,
    split_manifest_id: str | None = None,
) -> dict[str, Any]:
    writer = PrepackDatasetWriter(
        config=config,
        output_dir=output_dir,
        tokenizer_hash=tokenizer_hash,
        data_snapshot_hash=data_snapshot_hash,
        source_dataset_versions=source_dataset_versions,
        synthetic_repo_id=synthetic_repo_id,
        synthetic_manifest_id=synthetic_manifest_id,
        split_manifest_id=split_manifest_id,
    )
    sequences = tokenize_and_pack_examples(examples, tokenizer, config=config)
    writer.add_sequences(sequences, target_sequences=target_sequences)
    return writer.close()


def _requires_atomic_pack(example: TokenizedExample) -> bool:
    labels = [
        example.example_type,
        str(example.metadata.get("task_type", "")),
        str(example.metadata.get("training_stage", "")),
    ]
    if example.metadata.get("fim_applied"):
        return True
    text = " ".join(labels).lower()
    return any(
        marker in text
        for marker in (
            "fim",
            "instruction",
            "debug",
            "comment",
            "refactor",
            "test",
        )
    )


def _attribution(example: TokenizedExample) -> _Attribution:
    metadata = example.metadata
    source = str(metadata.get("source") or example.source)
    task_type = str(metadata.get("task_type") or example.example_type)
    training_stage = str(example.training_stage or metadata.get("training_stage") or "unknown")
    quality_tier = str(example.quality_tier or metadata.get("quality_tier") or "unknown")
    split_key = (
        metadata.get("source_candidate_id")
        or metadata.get("candidate_id")
        or metadata.get("stable_input_hash")
        or metadata.get("id")
        or metadata.get("record_id")
        or f"{source}:{task_type}:{hash_payload({'tokens': example.token_ids})}"
    )
    return _Attribution(
        source=source,
        task_type=task_type,
        training_stage=training_stage,
        quality_tier=quality_tier,
        split_key=str(split_key),
    )


def _build_sequence(
    token_ids: Sequence[int],
    attrs: Sequence[_Attribution],
    *,
    packing_mode: str,
    atomic_boundary_violations: int,
) -> PrepackedSequence:
    if len(token_ids) != len(attrs):
        raise ValueError("token attribution length must match token length")
    source_counts = Counter(attr.source for attr in attrs)
    task_type_counts = Counter(attr.task_type for attr in attrs)
    training_stage_counts = Counter(attr.training_stage for attr in attrs)
    quality_tier_counts = Counter(attr.quality_tier for attr in attrs)
    unique_examples = {(attr.source, attr.split_key) for attr in attrs}
    source_example_counts = Counter(source for source, _ in unique_examples)
    split_key_material = "|".join(sorted({attr.split_key for attr in attrs}))
    split_key = hash_payload({"split_keys": split_key_material, "tokens": list(token_ids)})
    return PrepackedSequence(
        token_ids=list(token_ids),
        split_key=split_key,
        source_counts=dict(source_counts),
        task_type_counts=dict(task_type_counts),
        training_stage_counts=dict(training_stage_counts),
        quality_tier_counts=dict(quality_tier_counts),
        packing_mode_counts={packing_mode: 1},
        source_example_counts=dict(source_example_counts),
        atomic_boundary_violations=atomic_boundary_violations,
    )


def _merge_sequence_counts(
    sequences: Iterable[PrepackedSequence],
    attribute: str,
) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for sequence in sequences:
        value = getattr(sequence, attribute)
        counter.update({str(key): int(count) for key, count in value.items()})
    return dict(counter)
