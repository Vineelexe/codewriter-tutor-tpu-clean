from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cw360.local.runtime import LocalRuntimeError, reports_to_json, run_local_smoke  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 12.5 local runtime smoke check.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompt", default="Write a tiny Python function.")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    if args.steps < 20 or args.steps > 100:
        raise SystemExit("local smoke script steps must be in the 20-100 acceptance range")
    try:
        reports = run_local_smoke(config_path=args.config, prompt=args.prompt, steps=args.steps)
    except LocalRuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(reports_to_json(reports))


if __name__ == "__main__":
    main()
