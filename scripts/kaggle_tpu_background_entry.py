from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kaggle Save Version / Run All entrypoint for background TPU relay training."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepacked-dir", default=None)
    parser.add_argument("--checkpoint-dir", default="/kaggle/working/checkpoints")
    parser.add_argument("--durable-dir", default=None)
    parser.add_argument("--next-trainer", default="sarang")
    args = parser.parse_args()

    setup_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "kaggle_tpu_setup_check.py"),
        "--config",
        args.config,
        "--checkpoint-dir",
        args.checkpoint_dir,
    ]
    train_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "kaggle_tpu_train.py"),
        "--config",
        args.config,
        "--checkpoint-dir",
        args.checkpoint_dir,
        "--next-trainer",
        args.next_trainer,
    ]
    if args.prepacked_dir:
        setup_cmd.extend(["--prepacked-dir", args.prepacked_dir])
        train_cmd.extend(["--prepacked-dir", args.prepacked_dir])
    if args.durable_dir:
        setup_cmd.extend(["--durable-dir", args.durable_dir])
        train_cmd.extend(["--durable-dir", args.durable_dir])

    subprocess.run(setup_cmd, check=True)
    subprocess.run(train_cmd, check=True)


if __name__ == "__main__":
    main()
