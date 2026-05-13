from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cw360.prepack.verify import verify_prepacked_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify TPU prepacked .npy shards.")
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args()

    report = verify_prepacked_dataset(args.input_dir)
    print(
        "verified "
        f"{report.shard_count} shards, {report.total_sequences} sequences, "
        f"{report.total_tokens_for_training} training tokens"
    )


if __name__ == "__main__":
    main()
