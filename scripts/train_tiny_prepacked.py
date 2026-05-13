from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cw360.config import load_config  # noqa: E402
from cw360.train.cpu_trainer import CPUTinyTrainer  # noqa: E402
from cw360.train.shard_dataloader import (  # noqa: E402
    PrepackedShardDataLoader,
    data_snapshot_from_prepacked_manifest,
    validate_prepacked_manifest_for_training,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tiny CPU training from prepacked shards.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepacked-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    prepacked_dir = Path(args.prepacked_dir)
    snapshot = data_snapshot_from_prepacked_manifest(prepacked_dir)
    manifest = PrepackedShardDataLoader(
        prepacked_dir,
        split="train",
        batch_size=int(config["micro_batch_size"]),
        drop_last=bool(config.get("drop_last", True)),
        static_shapes=bool(config.get("static_shapes", True)),
        shard_order_seed=int(config.get("shard_order_seed", 0)),
    ).manifest
    validate_prepacked_manifest_for_training(manifest)

    output_dir = Path(args.output_dir or _timestamped_output_dir("outputs/tiny_prepacked_train"))

    def train_factory(cursor):
        return iter(
            PrepackedShardDataLoader(
                prepacked_dir,
                split="train",
                batch_size=int(config["micro_batch_size"]),
                drop_last=bool(config.get("drop_last", True)),
                static_shapes=bool(config.get("static_shapes", True)),
                shard_order_seed=int(config.get("shard_order_seed", 0)),
                cursor=cursor,
            )
        )

    def val_factory(cursor):
        del cursor
        return iter(
            PrepackedShardDataLoader(
                prepacked_dir,
                split="val",
                batch_size=int(config["micro_batch_size"]),
                drop_last=bool(config.get("drop_last", True)),
                static_shapes=bool(config.get("static_shapes", True)),
                shard_order_seed=int(config.get("shard_order_seed", 0)),
            )
        )

    first_config = dict(config)
    first_config["max_steps"] = 1
    first = CPUTinyTrainer(
        config=first_config,
        output_dir=output_dir,
        batch_factory=train_factory,
        val_batch_factory=val_factory,
        data_snapshot=snapshot,
    )
    first_result = first.train()

    second_config = dict(config)
    second_config["max_steps"] = max(2, int(config.get("max_steps", 2)))
    second = CPUTinyTrainer(
        config=second_config,
        output_dir=output_dir,
        batch_factory=train_factory,
        val_batch_factory=val_factory,
        data_snapshot=snapshot,
        resume_checkpoint=first_result.checkpoints[-1],
    )
    second_result = second.train()
    print(
        json.dumps(
            {
                "saved": str(first_result.checkpoints[-1]),
                "resumed_from": str(first_result.checkpoints[-1]),
                "final_checkpoint": str(second_result.checkpoints[-1]),
                "final_step": second_result.state.step,
                "tokens_seen": second_result.state.tokens_seen,
                "sequences_seen": second_result.state.sequences_seen,
                "prepacked_dir": str(prepacked_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _timestamped_output_dir(base: str) -> str:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return str(Path(base) / stamp)


if __name__ == "__main__":
    main()
