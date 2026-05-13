from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cw360.local.export import export_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a trained CodeWriter-Tutor checkpoint for local CPU inference."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--allow-test-tokenizer",
        action="store_true",
        help="Include a tiny tokenizer switch for unit-test exports only.",
    )
    args = parser.parse_args()

    result = export_checkpoint(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        generation_config={
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "seed": args.seed,
            "use_cache": True,
            "device": "cpu",
        },
        allow_test_tokenizer=args.allow_test_tokenizer,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
