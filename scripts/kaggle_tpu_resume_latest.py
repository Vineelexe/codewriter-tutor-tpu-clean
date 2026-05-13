from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cw360.checkpoint import find_latest_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Locate the latest valid local relay checkpoint.")
    parser.add_argument("--checkpoint-dir", default="/kaggle/working/checkpoints")
    args = parser.parse_args()
    latest = find_latest_checkpoint(args.checkpoint_dir)
    print(
        json.dumps(
            {
                "found": latest is not None,
                "checkpoint": None if latest is None else str(latest.path),
                "step": None if latest is None else latest.metadata.step,
                "tokens_seen": None if latest is None else latest.metadata.tokens_seen,
                "tpu_data_cursor": (
                    None
                    if latest is None or latest.metadata.tpu_data_cursor is None
                    else latest.metadata.tpu_data_cursor.to_dict()
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
