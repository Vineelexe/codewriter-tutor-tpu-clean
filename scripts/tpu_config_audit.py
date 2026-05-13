from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cw360.config import load_model_config  # noqa: E402
from cw360.model.count import build_parameter_report  # noqa: E402
from cw360.train.validate import validate_tpu_train_config  # noqa: E402


def audit_config(config_path: str | Path) -> dict[str, object]:
    config = validate_tpu_train_config(config_path)
    model_path = _resolve_model_path(config_path, str(config["model_config_path"]))
    model_config = load_model_config(model_path)
    parameter_report = build_parameter_report(model_config)
    return {
        "ok": True,
        "config": str(config_path),
        "model_config_path": str(config["model_config_path"]),
        "size_label": model_config.size_label,
        "parameter_count": parameter_report.total_parameters,
        "tokenizer_id": str(config["tokenizer_id"]),
        "training_stage": str(config["training_stage"]),
        "platform": str(config["platform"]),
        "max_seq_len": int(config["max_seq_len"]),
        "drop_last": bool(config["drop_last"]),
        "dynamic_padding": bool(config["dynamic_padding"]),
        "precision": str(config["precision"]),
        "prepacked_manifest_path": str(config["prepacked_manifest_path"]),
        "later_stage_only": bool(config.get("later_stage_only", False)),
    }


def _resolve_model_path(config_path: str | Path, model_config_path: str) -> Path:
    model_path = Path(model_config_path)
    if model_path.is_absolute():
        return model_path
    path = Path(config_path).resolve()
    root = path.parents[1] if path.parent.name == "configs" else ROOT
    return root / model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a real TPU train config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = audit_config(args.config)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("TPU config audit passed")
        for key, value in report.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
