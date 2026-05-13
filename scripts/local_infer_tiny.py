from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cw360.local.runtime import LocalRuntimeError, run_tiny_inference_check  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tiny local CPU inference.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    try:
        report = run_tiny_inference_check(
            args.config,
            prompt=args.prompt,
            checkpoint_path=args.checkpoint,
            max_new_tokens=args.max_new_tokens,
        )
    except LocalRuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
