from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cw360.checkpoint import find_latest_checkpoint, write_handoff_manifest  # noqa: E402
from cw360.config import load_config  # noqa: E402
from cw360.kaggle import build_durable_store_from_config  # noqa: E402
from cw360.train.tpu_trainer import TPUTrainer  # noqa: E402
from cw360.train.xla_utils import TorchXLANotAvailable  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Kaggle TPU relay training from prepacked shards."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepacked-dir", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume", default="latest")
    parser.add_argument("--durable-dir", default=None)
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--next-trainer", default="sarang")
    args = parser.parse_args()

    config = load_config(args.config)
    prepacked_dir = Path(args.prepacked_dir or config["prepacked_shards_path"])
    checkpoint_dir = Path(
        args.checkpoint_dir or config.get("checkpoint_dir", "/kaggle/working/checkpoints")
    )
    resume = _resolve_resume(args.resume, checkpoint_dir)
    durable_store = build_durable_store_from_config(config, durable_dir=args.durable_dir)
    uploaded: list[dict[str, object]] = []

    def on_checkpoint(path: Path) -> None:
        if args.no_upload:
            return
        uploaded.append(durable_store.upload_checkpoint(path).to_dict())

    trainer = TPUTrainer(
        config=config,
        prepacked_dir=prepacked_dir,
        output_dir=checkpoint_dir,
        resume_checkpoint=resume,
        checkpoint_callback=on_checkpoint,
    )
    try:
        result = trainer.train()
    except TorchXLANotAvailable as exc:
        raise SystemExit(str(exc)) from exc

    latest = result.checkpoints[-1] if result.checkpoints else None
    handoff_path = None
    if latest is not None:
        handoff_path = write_handoff_manifest(
            checkpoint_path=latest,
            trainer_leaving=str(config.get("trained_by", "vineel")),
            next_trainer_expected=args.next_trainer,
            exact_resume_command=(
                "python scripts/kaggle_tpu_train.py "
                f"--config {args.config} --resume {latest}"
            ),
        )
    print(
        json.dumps(
            {
                "final_step": result.state.step,
                "checkpoints": [str(path) for path in result.checkpoints],
                "resumed_from": None if result.resumed_from is None else str(result.resumed_from),
                "handoff_manifest": None if handoff_path is None else str(handoff_path),
                "uploaded": uploaded,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _resolve_resume(value: str | None, checkpoint_dir: Path) -> Path | None:
    if value in (None, "", "none"):
        return None
    if value == "latest":
        latest = find_latest_checkpoint(checkpoint_dir)
        return None if latest is None else latest.path
    return Path(value)


if __name__ == "__main__":
    main()
