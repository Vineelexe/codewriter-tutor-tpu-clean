from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cw360.config import load_model_config  # noqa: E402
from cw360.model.count import build_parameter_report  # noqa: E402
from cw360.prepack.manifest import read_json  # noqa: E402
from cw360.prepack.shard_index import grouped_shard_paths  # noqa: E402
from cw360.train.validate import validate_tpu_train_config  # noqa: E402


def build_dry_run_summary(
    config_path: str | Path,
    *,
    no_tpu_required: bool = False,
) -> dict[str, Any]:
    config = validate_tpu_train_config(config_path)
    model_config = load_model_config(
        _resolve_model_path(config_path, str(config["model_config_path"]))
    )
    parameter_report = build_parameter_report(model_config)

    manifest_path = Path(str(config["prepacked_manifest_path"]))
    manifest = _load_manifest_if_available(manifest_path)
    batch_summary = _inspect_one_batch_if_available(
        manifest=manifest,
        input_dir=Path(str(config["prepacked_shards_path"])),
        batch_size=int(config["fixed_batch_size"]),
        max_seq_len=int(config["max_seq_len"]),
    )
    checkpoint_status = _verify_checkpoint_path(
        Path(str(config["checkpoint_dir"])),
        no_tpu_required=no_tpu_required,
    )

    return {
        "ok": True,
        "config": str(config_path),
        "model_config_path": str(config["model_config_path"]),
        "size_label": model_config.size_label,
        "parameter_count": parameter_report.total_parameters,
        "manifest": _manifest_summary(manifest_path, manifest),
        "batch": batch_summary,
        "fixed_shapes": {
            "max_seq_len": int(config["max_seq_len"]),
            "seq_len_plus_one": int(config["max_seq_len"]) + 1,
            "batch_size": int(config["fixed_batch_size"]),
            "drop_last": bool(config["drop_last"]),
            "dynamic_padding": bool(config["dynamic_padding"]),
        },
        "checkpoint_path": checkpoint_status,
        "durable_checkpoint_store": {
            "configured": True,
            "type": config["durable_checkpoint_store"]["type"],
            "path_prefix": config["durable_checkpoint_store"]["path_prefix"],
            "private": config["durable_checkpoint_store"]["private"],
        },
        "time_guards": {
            "save_every_n_minutes": int(config["save_every_n_minutes"]),
            "session_time_limit_minutes": int(config["session_time_limit_minutes"]),
            "save_before_exit_minutes": int(config["save_before_exit_minutes"]),
        },
    }


def _load_manifest_if_available(path: Path) -> dict[str, Any] | None:
    return read_json(path) if path.exists() else None


def _resolve_model_path(config_path: str | Path, model_config_path: str) -> Path:
    model_path = Path(model_config_path)
    if model_path.is_absolute():
        return model_path
    path = Path(config_path).resolve()
    root = path.parents[1] if path.parent.name == "configs" else ROOT
    return root / model_path


def _manifest_summary(path: Path, manifest: dict[str, Any] | None) -> dict[str, Any]:
    if manifest is None:
        return {
            "path": str(path),
            "available": False,
            "note": "manifest path is configured but not present in this local environment",
        }
    return {
        "path": str(path),
        "available": True,
        "manifest_hash": manifest.get("manifest_hash"),
        "shard_manifest_hash": manifest.get("shard_manifest_hash"),
        "max_seq_len": manifest.get("max_seq_len"),
        "seq_len_plus_one": manifest.get("seq_len_plus_one"),
        "dtype": manifest.get("dtype"),
    }


def _inspect_one_batch_if_available(
    *,
    manifest: dict[str, Any] | None,
    input_dir: Path,
    batch_size: int,
    max_seq_len: int,
) -> dict[str, Any]:
    if manifest is None:
        return {"available": False, "note": "no local manifest available for batch inspection"}
    shard_paths = grouped_shard_paths(manifest, input_dir).get("train", [])
    if not shard_paths:
        return {"available": False, "note": "manifest contains no train shards"}
    shard = np.load(shard_paths[0], mmap_mode="r")
    if shard.ndim != 2:
        raise ValueError("prepacked shard must be rank 2")
    if shard.dtype != np.uint16:
        raise ValueError("prepacked shard dtype must be uint16")
    if int(shard.shape[1]) != max_seq_len + 1:
        raise ValueError("prepacked shard shape must be [n, max_seq_len + 1]")
    if int(shard.shape[0]) < batch_size:
        return {
            "available": False,
            "note": "train shard exists but has fewer rows than fixed batch size",
            "first_shard_shape": [int(shard.shape[0]), int(shard.shape[1])],
        }
    input_shape = [batch_size, max_seq_len]
    labels_shape = [batch_size, max_seq_len]
    return {
        "available": True,
        "first_shard": str(shard_paths[0]),
        "first_shard_shape": [int(shard.shape[0]), int(shard.shape[1])],
        "input_ids_shape": input_shape,
        "labels_shape": labels_shape,
    }


def _verify_checkpoint_path(path: Path, *, no_tpu_required: bool) -> dict[str, Any]:
    if no_tpu_required and path.as_posix().startswith("/kaggle/"):
        return {
            "path": str(path),
            "writable": None,
            "note": "skipped local write check for Kaggle-only checkpoint path",
        }
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".dry_run_write_check"
    probe.write_text("ok\n", encoding="utf-8")
    probe.unlink()
    return {"path": str(path), "writable": True}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dry-run TPU train startup without a large forward pass."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-tpu-required", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = build_dry_run_summary(args.config, no_tpu_required=args.no_tpu_required)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print("train dry-run summary")
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
