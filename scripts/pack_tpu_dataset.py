from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cw360.data.modes import DataPipelineMode  # noqa: E402
from cw360.data.pipeline import (  # noqa: E402
    DataPipeline,
    PipelineOptions,
    source_iterators_from_dataset_configs,
)
from cw360.data.sources import DatasetSourceConfig, load_dataset_source_configs  # noqa: E402
from cw360.prepack.config import load_prepack_config  # noqa: E402
from cw360.prepack.manifest import hash_payload  # noqa: E402
from cw360.prepack.writer import (  # noqa: E402
    PrepackDatasetWriter,
    tokenize_and_pack_examples,
)
from cw360.tokenizer.loader import load_tokenizer  # noqa: E402
from cw360.tokenizer.validate import assert_uint16_safe, tokenizer_vocab  # noqa: E402

SOURCE_ALIASES = {
    "debugbench_and_structured_debugging": "debugbench",
    "structured_synthetic_instructions": "synthetic",
    "cosmopedia_cs_algorithms": "cosmopedia_cs",
}


@dataclass(frozen=True, slots=True)
class PackTpuDatasetResult:
    manifest: dict[str, Any]
    sequences_written: int
    stop_reason: str
    output_dir: str


def pack_tpu_dataset(
    *,
    config_path: str | Path,
    datasets_config_path: str | Path,
    synthetic_input: str | Path | None = None,
    output_dir: str | Path | None = None,
    target_sequences: int | None = None,
    training_stage: str = "base_pretrain",
    stop_after_seconds: float | None = None,
) -> PackTpuDatasetResult:
    config = load_prepack_config(config_path)
    if output_dir:
        config = config.with_overrides(output_dir=output_dir)

    tokenizer = load_tokenizer(config.prepack.tokenizer_id)
    assert_uint16_safe(tokenizer)
    tokenizer_hash = hash_payload(
        {
            "tokenizer_id": config.prepack.tokenizer_id,
            "vocab": tokenizer_vocab(tokenizer),
        }
    )

    source_configs = _load_source_configs(
        str(datasets_config_path),
        synthetic_input=None if synthetic_input is None else str(synthetic_input),
    )
    selected_weights = _selected_weights(config.mixture, source_configs)
    config = replace(config, mixture=selected_weights)
    source_iterators = source_iterators_from_dataset_configs(
        source_configs,
        source_names=selected_weights.keys(),
    )
    pipeline = DataPipeline(
        source_iterators,
        PipelineOptions(
            mode=DataPipelineMode.PREPACK_STREAM,
            training_stage=training_stage,
            seed=config.prepack.seed,
            max_seq_len=config.prepack.max_seq_len,
            allow_exhausted_sources=True,
        ),
        tokenizer=tokenizer,
        weights=selected_weights,
    )
    data_snapshot_hash = hash_payload(
        {
            "datasets_config": str(Path(datasets_config_path).resolve()),
            "synthetic_input": None if synthetic_input is None else str(synthetic_input),
            "weights": selected_weights,
        }
    )
    writer = PrepackDatasetWriter(
        config=config,
        output_dir=output_dir,
        tokenizer_hash=tokenizer_hash,
        data_snapshot_hash=data_snapshot_hash,
        source_dataset_versions=_source_dataset_versions(source_configs),
        synthetic_repo_id="phase-11-validated-synthetic",
        synthetic_manifest_id=Path(synthetic_input).name if synthetic_input else None,
        split_manifest_id=f"stable_hash:{config.splits.synthetic_split_key}",
    )
    sequence_iter = iter(
        tokenize_and_pack_examples(
            pipeline.prepack_stream(),
            tokenizer,
            config=config,
        )
    )
    deadline = (
        time.monotonic() + float(stop_after_seconds)
        if stop_after_seconds is not None
        else None
    )
    written = 0
    stop_reason = "data_exhausted"
    while True:
        if target_sequences is not None and written >= target_sequences:
            stop_reason = "target_sequences"
            break
        if deadline is not None and time.monotonic() >= deadline:
            stop_reason = "time_guard"
            break
        try:
            sequence = next(sequence_iter)
        except StopIteration:
            stop_reason = "data_exhausted"
            break
        if deadline is not None and time.monotonic() >= deadline:
            stop_reason = "time_guard"
            break
        writer.add_sequence(sequence)
        written += 1

    manifest = writer.close()
    return PackTpuDatasetResult(
        manifest=manifest,
        sequences_written=written,
        stop_reason=stop_reason,
        output_dir=str(config.prepack.output_dir),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack final TPU .npy token shards on CPU.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--datasets-config", required=True)
    parser.add_argument("--synthetic-input")
    parser.add_argument("--output-dir")
    parser.add_argument("--target-sequences", type=int)
    parser.add_argument("--training-stage", default="base_pretrain")
    args = parser.parse_args()

    result = pack_tpu_dataset(
        config_path=args.config,
        datasets_config_path=args.datasets_config,
        synthetic_input=args.synthetic_input,
        output_dir=args.output_dir,
        target_sequences=args.target_sequences,
        training_stage=args.training_stage,
    )
    print(
        "wrote TPU prepacked dataset "
        f"to {result.output_dir} with {result.manifest['total_sequences']} sequences "
        f"(stop_reason={result.stop_reason})"
    )


def _load_source_configs(
    datasets_config: str,
    *,
    synthetic_input: str | None,
) -> dict[str, DatasetSourceConfig]:
    configs = load_dataset_source_configs(datasets_config)
    if synthetic_input:
        synthetic = configs.get("synthetic")
        if synthetic is None:
            synthetic = DatasetSourceConfig(name="synthetic", extractor="synthetic")
        configs["synthetic"] = replace(synthetic, local_path=synthetic_input)
    return configs


def _selected_weights(
    mixture: dict[str, float],
    source_configs: dict[str, DatasetSourceConfig],
) -> dict[str, float]:
    selected: dict[str, float] = {}
    for name, weight in mixture.items():
        source_name = SOURCE_ALIASES.get(name, name)
        if source_name in source_configs and weight > 0:
            selected[source_name] = selected.get(source_name, 0.0) + float(weight)
    if not selected:
        raise ValueError("no configured prepack mixture sources are available")
    return selected


def _source_dataset_versions(configs: dict[str, DatasetSourceConfig]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name, config in configs.items():
        identity: dict[str, Any] = {
            "dataset_id": config.dataset_id,
            "subset": config.subset,
            "split": config.split,
            "local_path": config.local_path,
        }
        versions[name] = hash_payload(identity)
    return versions


if __name__ == "__main__":
    main()
