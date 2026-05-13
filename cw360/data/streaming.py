from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from typing import Any

from cw360.data.extractors import extract_training_example
from cw360.data.schemas import ExtractedRecord, TrainingExample
from cw360.data.sources import DatasetSourceConfig
from cw360.data.synthetic import iter_synthetic_jsonl


class DatasetAccessError(RuntimeError):
    pass


def _load_hf_dataset(config: DatasetSourceConfig) -> Iterator[Mapping[str, Any]]:
    if not config.dataset_id:
        raise DatasetAccessError(f"source {config.name} has no dataset_id")
    token = os.getenv("HF_TOKEN") or None
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise DatasetAccessError("install datasets to stream Hugging Face sources") from exc
    try:
        dataset = load_dataset(
            config.dataset_id,
            config.subset,
            split=config.split,
            streaming=config.streaming,
            token=token,
        )
    except Exception as exc:
        hint = "Set HF_TOKEN or run huggingface-cli login if this source is gated."
        raise DatasetAccessError(
            f"could not stream {config.name} from Hugging Face: {exc}. {hint}"
        ) from exc
    return iter(dataset)


def iter_source_records(config: DatasetSourceConfig) -> Iterator[ExtractedRecord]:
    if config.name == "synthetic" or config.local_path:
        if not config.local_path:
            raise DatasetAccessError("synthetic source requires local_path")
        yield from iter_synthetic_jsonl(config.local_path)
        return

    extractor = config.extractor or config.name
    for row in _load_hf_dataset(config):
        try:
            example = extract_training_example(extractor, row)
        except (TypeError, ValueError) as exc:
            yield ExtractedRecord(None, error=f"malformed source row: {exc}")
            continue
        yield ExtractedRecord(example)


def iter_training_examples(config: DatasetSourceConfig) -> Iterator[TrainingExample]:
    for record in iter_source_records(config):
        if record.example is not None:
            yield record.example
