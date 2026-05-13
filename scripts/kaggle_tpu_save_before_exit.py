from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cw360.checkpoint import find_latest_checkpoint  # noqa: E402
from cw360.config import load_config  # noqa: E402
from cw360.kaggle import build_durable_store_from_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload the latest checkpoint before a Kaggle exit.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", default="/kaggle/working/checkpoints")
    parser.add_argument("--durable-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    latest = find_latest_checkpoint(args.checkpoint_dir)
    if latest is None:
        raise SystemExit(f"no valid checkpoint found in {args.checkpoint_dir}")
    store = build_durable_store_from_config(config, durable_dir=args.durable_dir)
    record = store.upload_checkpoint(latest.path, dry_run=args.dry_run)
    print(json.dumps(record.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
