from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cw360.config import load_model_config  # noqa: E402
from cw360.local.memory import build_local_memory_report  # noqa: E402
from cw360.local.runtime import parameter_report_to_dict, run_full_config_count_check  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Report local CPU memory estimates.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--no-forward",
        action="store_true",
        help="Required for full candidate configs.",
    )
    args = parser.parse_args()

    config = load_model_config(args.config)
    if config.size_label != "tiny" and not args.no_forward:
        raise SystemExit("full candidate configs require --no-forward for local memory checks")
    parameter_report = run_full_config_count_check(args.config, no_forward=True)
    memory_report = build_local_memory_report(config)
    print(
        json.dumps(
            {
                "parameter_report": parameter_report_to_dict(parameter_report),
                "memory_report": memory_report.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
