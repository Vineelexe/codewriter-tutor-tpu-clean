from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cw360.prepack.config import PrepackFactoryConfig, load_prepack_config  # noqa: E402
from cw360.prepack.manifest import hash_payload  # noqa: E402
from cw360.prepack.shard_index import SPLITS, SplitName  # noqa: E402
from cw360.prepack.writer import PrepackDatasetWriter, PrepackedSequence  # noqa: E402


def make_tiny_prepacked_dataset(
    *,
    config: PrepackFactoryConfig,
    output_dir: str | Path,
    num_sequences: int,
) -> dict[str, object]:
    if num_sequences <= 0:
        raise ValueError("num_sequences must be positive")
    tiny_config = config.with_overrides(
        output_dir=output_dir,
        shard_num_sequences=num_sequences,
        rolling_shuffle_buffer_sequences=max(1, min(num_sequences, 64)),
        minimum_shuffle_buffer_sequences=max(1, min(num_sequences, 64)),
    )
    writer = PrepackDatasetWriter(
        config=tiny_config,
        output_dir=output_dir,
        tokenizer_hash=hash_payload({"tokenizer": "tiny-deterministic-tokenizer"}),
        data_snapshot_hash=hash_payload({"source": "tiny-synthetic-prepacked", "n": num_sequences}),
        source_dataset_versions={"tiny_synthetic": "phase-11.5-local"},
        synthetic_repo_id="local-tiny-synthetic",
        synthetic_manifest_id="tiny-prepacked-v1",
        split_manifest_id="tiny-stable-split-v1",
        code_git_commit=None,
    )
    for split_index, split in enumerate(SPLITS):
        for sequence_index, source in enumerate(_source_schedule(config.mixture, num_sequences)):
            writer.add_sequence(
                _tiny_sequence(
                    config=tiny_config,
                    split=split,
                    split_index=split_index,
                    sequence_index=sequence_index,
                    source=source,
                ),
                split=split,
            )
    return writer.close()


def _source_schedule(mixture: Mapping[str, float], num_sequences: int) -> list[str]:
    target = {name: float(weight) for name, weight in mixture.items() if weight > 0}
    total = sum(target.values())
    raw_counts = {name: (weight / total) * num_sequences for name, weight in target.items()}
    counts = {name: int(value) for name, value in raw_counts.items()}
    remaining = num_sequences - sum(counts.values())
    remainders = sorted(
        ((raw_counts[name] - counts[name], name) for name in target),
        reverse=True,
    )
    for _, name in remainders[:remaining]:
        counts[name] += 1
    schedule: list[str] = []
    for name in sorted(counts):
        schedule.extend([name] * counts[name])
    return schedule


def _tiny_sequence(
    *,
    config: PrepackFactoryConfig,
    split: SplitName,
    split_index: int,
    sequence_index: int,
    source: str,
) -> PrepackedSequence:
    seq_len = config.prepack.seq_len_plus_one
    base = 17 + split_index * 1000 + sequence_index * 7
    token_ids = ((np.arange(seq_len, dtype=np.uint32) + base) % 49_152).astype(np.uint16)
    task_type = (
        "python_code"
        if source in {"stack_v2_python", "codesearchnet_python"}
        else "instruction"
    )
    training_stage = "base_pretrain" if task_type == "python_code" else "instruction_tune"
    quality_tier = "A" if sequence_index % 2 == 0 else "B"
    packing_mode = (
        "continuous_pack" if task_type == "python_code" else "example_atomic_pack"
    )
    return PrepackedSequence(
        token_ids=token_ids,
        split_key=f"tiny:{split}:{source}:{sequence_index}",
        source_counts={source: 1},
        task_type_counts={task_type: 1},
        training_stage_counts={training_stage: 1},
        quality_tier_counts={quality_tier: 1},
        packing_mode_counts={packing_mode: 1},
        source_example_counts={source: 1},
        atomic_boundary_violations=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny local TPU prepacked dataset.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-sequences", type=int, default=128)
    args = parser.parse_args()

    config = load_prepack_config(args.config)
    manifest = make_tiny_prepacked_dataset(
        config=config,
        output_dir=args.output_dir,
        num_sequences=args.num_sequences,
    )
    print(
        "wrote tiny prepacked dataset "
        f"to {args.output_dir} with {manifest['total_sequences']} total sequences"
    )


if __name__ == "__main__":
    main()
