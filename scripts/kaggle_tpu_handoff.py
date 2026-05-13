from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cw360.checkpoint import build_handoff_manifest, write_handoff_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a relay handoff manifest for a checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trainer-leaving", required=True, choices=["vineel", "sarang"])
    parser.add_argument("--next-trainer", required=True, choices=["vineel", "sarang"])
    parser.add_argument("--resume-command", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    output = write_handoff_manifest(
        checkpoint_path=args.checkpoint,
        trainer_leaving=args.trainer_leaving,
        next_trainer_expected=args.next_trainer,
        exact_resume_command=args.resume_command,
        output_path=args.output,
    )
    manifest = build_handoff_manifest(
        checkpoint_path=args.checkpoint,
        trainer_leaving=args.trainer_leaving,
        next_trainer_expected=args.next_trainer,
        exact_resume_command=args.resume_command,
    )
    print(json.dumps({"handoff_manifest": str(output), **manifest.to_dict()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
