from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cw360.config import load_model_config  # noqa: E402
from cw360.model.count import build_parameter_report, validate_size_label_count  # noqa: E402
from cw360.model.lm import CodeWriterTutorLM  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Count CodeWriter-Tutor LM parameters.")
    parser.add_argument("--config", type=Path, required=True, help="Path to model YAML config")
    parser.add_argument(
        "--no-forward",
        action="store_true",
        help="Use the config formula only; do not instantiate or forward the model.",
    )
    args = parser.parse_args()

    config = load_model_config(args.config)
    model = None if args.no_forward else CodeWriterTutorLM(config)
    report = build_parameter_report(config, model=model)
    payload = {
        "size_label": report.size_label,
        "total_parameters": report.total_parameters,
        "trainable_parameters": report.trainable_parameters,
        "embedding_parameters": report.embedding_parameters,
        "per_block_estimate": report.per_block_estimate,
        "expected_parameters": report.expected_parameters,
        "actual_parameters": report.actual_parameters,
        "delta_from_expected": report.delta_from_expected,
        "relative_delta_from_expected": report.relative_delta_from_expected,
        "size_label_valid": report.size_label_valid,
        "within_label_tolerance": validate_size_label_count(report),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["within_label_tolerance"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
