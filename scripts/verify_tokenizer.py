from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cw360.tokenizer.loader import TokenizerLoadError, tokenizer_id_from_config  # noqa: E402
from cw360.tokenizer.validate import validate_tokenizer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the configured StarCoder2 tokenizer.")
    parser.add_argument("--config", type=Path, default=None, help="Model config YAML path")
    parser.add_argument(
        "--tokenizer-id",
        default=None,
        help="Override tokenizer id; defaults to config tokenizer_name or the project default",
    )
    parser.add_argument(
        "--require-pad-for-batching",
        action="store_true",
        help="Set pad_token=eos_token for validation contexts that require batching.",
    )
    args = parser.parse_args()

    tokenizer_id = args.tokenizer_id or tokenizer_id_from_config(args.config)
    try:
        report = validate_tokenizer(
            tokenizer_id,
            require_pad_for_batching=args.require_pad_for_batching,
        )
    except TokenizerLoadError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"tokenizer validation failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
