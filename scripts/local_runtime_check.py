from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cw360.local.runtime import (  # noqa: E402
    LocalRuntimeError,
    parameter_report_to_dict,
    run_full_config_count_check,
    run_synthetic_validation_check,
    run_teacher_mock_check,
    run_tiny_model_step_check,
    run_tiny_prepacked_train_resume_check,
    run_tiny_synthetic_train_resume_check,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run targeted local runtime checks.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "tiny-step",
            "teacher-mock",
            "synthetic-validation",
            "train-tiny",
            "prepack-tiny",
            "count",
        ],
    )
    parser.add_argument("--config", type=Path, default=Path("configs/model_tiny.yaml"))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--no-forward", action="store_true")
    args = parser.parse_args()

    try:
        payload = _run(args)
    except LocalRuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(payload, indent=2, sort_keys=True))


def _run(args: argparse.Namespace) -> dict[str, object]:
    if args.mode == "tiny-step":
        return run_tiny_model_step_check(args.config).to_dict()
    if args.mode == "teacher-mock":
        return run_teacher_mock_check().to_dict()
    if args.mode == "synthetic-validation":
        return run_synthetic_validation_check().to_dict()
    if args.mode == "count":
        return parameter_report_to_dict(
            run_full_config_count_check(args.config, no_forward=args.no_forward)
        )
    if args.mode == "train-tiny":
        with tempfile.TemporaryDirectory(prefix="cw360-local-train-") as tmp:
            return run_tiny_synthetic_train_resume_check(
                model_config_path=args.config,
                output_dir=Path(tmp) / "ckpts",
                steps=args.steps,
            ).to_dict()
    if args.mode == "prepack-tiny":
        with tempfile.TemporaryDirectory(prefix="cw360-local-prepack-") as tmp:
            root = Path(tmp)
            return run_tiny_prepacked_train_resume_check(
                model_config_path=args.config,
                output_dir=root / "ckpts",
                prepacked_dir=root / "prepacked",
                steps=args.steps,
            ).to_dict()
    raise AssertionError(f"unhandled mode: {args.mode}")


if __name__ == "__main__":
    main()
