from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cw360.eval.runner import EvalRunConfig, run_evaluation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the random-model fixed eval baseline.")
    parser.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help="Path to model YAML config",
    )
    parser.add_argument("--eval-config", type=Path, required=True, help="Path to eval YAML config")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eval/random_baseline"),
        help="Directory for random baseline JSONL and Markdown outputs",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override generated token count",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override generation temperature",
    )
    args = parser.parse_args()

    config = EvalRunConfig.from_eval_config(
        args.eval_config,
        model_config_path=args.model_config,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        random_baseline=True,
    )
    jsonl_path, markdown_path = run_evaluation(config)
    print(f"wrote {jsonl_path}")
    print(f"wrote {markdown_path}")


if __name__ == "__main__":
    main()
