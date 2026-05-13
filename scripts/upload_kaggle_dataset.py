from __future__ import annotations

import argparse
from pathlib import Path

from cw360.prepack.kaggle_dataset import run_kaggle_dataset_upload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create or update a private Kaggle Dataset from prepacked shards."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--create", action="store_true", help="Create instead of versioning.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not (input_dir / "dataset-metadata.json").exists():
        raise SystemExit("dataset-metadata.json is required before Kaggle upload")
    result = run_kaggle_dataset_upload(input_dir, create=args.create, private=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
