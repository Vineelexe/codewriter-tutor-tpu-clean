from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cw360.train.tpu_trainer import TPUTrainer  # noqa: E402
from cw360.train.xla_utils import TorchXLANotAvailable  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TPU/XLA training from prepacked shards.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepacked-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    trainer = TPUTrainer(
        config=args.config,
        prepacked_dir=args.prepacked_dir,
        output_dir=args.output_dir,
        resume_checkpoint=args.resume,
    )
    try:
        trainer.train()
    except TorchXLANotAvailable as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
