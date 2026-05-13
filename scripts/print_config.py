from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cw360.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and print a YAML config.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    print(json.dumps(config, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
