from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cw360.checkpoint import metadata_path_for_checkpoint, read_metadata_json  # noqa: E402
from cw360.kaggle import verify_checkpoint_bundle  # noqa: E402
from cw360.utils.hashing import sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a relay checkpoint and metadata bundle.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-sha256", default=None)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    verify_checkpoint_bundle(checkpoint)
    actual_sha = sha256_file(checkpoint)
    if args.expected_sha256 and args.expected_sha256 != actual_sha:
        raise SystemExit(
            f"checkpoint sha256 mismatch: expected={args.expected_sha256} actual={actual_sha}"
        )
    metadata = read_metadata_json(metadata_path_for_checkpoint(checkpoint))
    print(
        json.dumps(
            {
                "ok": True,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": actual_sha,
                "metadata": str(metadata_path_for_checkpoint(checkpoint)),
                "metadata_sha256": sha256_file(metadata_path_for_checkpoint(checkpoint)),
                "step": metadata.step,
                "tokens_seen": metadata.tokens_seen,
                "tpu_data_cursor": (
                    None if metadata.tpu_data_cursor is None else metadata.tpu_data_cursor.to_dict()
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
