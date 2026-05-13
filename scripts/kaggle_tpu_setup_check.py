from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from cw360.config import load_config, load_model_config  # noqa: E402
from cw360.kaggle import build_durable_store_from_config  # noqa: E402
from cw360.tokenizer.loader import load_tokenizer  # noqa: E402
from cw360.tokenizer.validate import validate_loaded_tokenizer  # noqa: E402


def build_setup_report(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    model_config = load_model_config(config["model_config_path"])
    prepacked_dir = Path(args.prepacked_dir or config.get("prepacked_shards_path", ""))
    manifest_path = prepacked_dir / "manifest.json"
    checkpoint_dir = Path(args.checkpoint_dir)
    durable_store = build_durable_store_from_config(config, durable_dir=args.durable_dir)
    tokenizer_status = _tokenizer_status(model_config.tokenizer_name, dry_run=bool(args.dry_run))
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_xla_available": importlib.util.find_spec("torch_xla") is not None,
        "xla_device_available": _xla_device_available(),
        "tpu_device_count": _tpu_device_count(),
        "hf_token_present": bool(os.environ.get("HF_TOKEN")),
        "tokenizer_load_check": tokenizer_status,
        "prepacked_kaggle_dataset_path_exists": prepacked_dir.exists(),
        "prepacked_dataset_path": str(prepacked_dir),
        "prepacked_manifest_exists": manifest_path.exists(),
        "prepacked_manifest_path": str(manifest_path),
        "writable_checkpoint_directory": _writable_status(checkpoint_dir, dry_run=bool(args.dry_run)),
        "durable_storage_target_status": durable_store.status(),
        "session_time_limit_minutes": int(config.get("session_time_limit_minutes", 510)),
        "quota_reminder": (
            "Verify Kaggle's current TPU quota and session limits from the account UI/docs "
            "before launch; do not rely on stale runtime claims."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Kaggle TPU relay training readiness.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepacked-dir", default=None)
    parser.add_argument("--checkpoint-dir", default="/kaggle/working/checkpoints")
    parser.add_argument("--durable-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = build_setup_report(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not args.dry_run and not _critical_checks_pass(report):
        raise SystemExit(1)


def _tokenizer_status(tokenizer_id: str, *, dry_run: bool) -> dict[str, Any]:
    try:
        tokenizer = load_tokenizer(tokenizer_id, local_files_only=dry_run)
        report = validate_loaded_tokenizer(tokenizer, tokenizer_id)
    except Exception as exc:
        return {
            "ok": False,
            "tokenizer_id": tokenizer_id,
            "error": str(exc),
            "dry_run_nonfatal": dry_run,
        }
    return {
        "ok": True,
        "tokenizer_id": tokenizer_id,
        "vocab_size": report.vocab_size,
        "max_token_id": report.max_token_id,
        "fim_tokens_found": report.fim_tokens,
    }


def _xla_device_available() -> bool:
    try:
        import torch_xla.core.xla_model as xm

        xm.xla_device()
    except Exception:
        return False
    return True


def _tpu_device_count() -> int | None:
    try:
        import torch_xla.core.xla_model as xm

        return int(xm.xrt_world_size())
    except Exception:
        return None


def _writable_status(path: Path, *, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {"path": str(path), "checked": False, "ok": path.exists()}
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".cw360_write_check"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return {"path": str(path), "checked": True, "ok": False, "error": str(exc)}
    return {"path": str(path), "checked": True, "ok": True}


def _critical_checks_pass(report: dict[str, Any]) -> bool:
    return bool(
        report["torch_xla_available"]
        and report["xla_device_available"]
        and report["tokenizer_load_check"]["ok"]
        and report["prepacked_kaggle_dataset_path_exists"]
        and report["prepacked_manifest_exists"]
        and report["writable_checkpoint_directory"]["ok"]
        and _durable_ready(report["durable_storage_target_status"])
    )


def _durable_ready(status: dict[str, Any]) -> bool:
    if status.get("type") == "hf_model_repo":
        return bool(status.get("repo_id_configured") and status.get("token_present"))
    if status.get("type") == "local_dir":
        return bool(status.get("path"))
    return False


if __name__ == "__main__":
    main()
