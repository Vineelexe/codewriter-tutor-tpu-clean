from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cw360.prepack.manifest import utc_now_iso, write_json  # noqa: E402
from cw360.prepack.session_runner import (  # noqa: E402
    SESSION_INFO_FILENAME,
    ensure_session_dir_available,
    next_session_id,
    scan_prepack_sessions,
    target_is_complete,
)
from scripts.pack_tpu_dataset import pack_tpu_dataset  # noqa: E402

SAFETY_BUFFER_SECONDS = 5 * 60


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one safe repeatable Kaggle CPU prepack session."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--datasets-config", required=True)
    parser.add_argument("--synthetic-input")
    parser.add_argument("--sessions-root", default="outputs/tpu_shards_2048_sessions")
    parser.add_argument("--session-id")
    parser.add_argument("--target-total-sequences", type=int, required=True)
    parser.add_argument("--max-session-hours", type=float, default=11.0)
    parser.add_argument("--session-target-sequences", type=int)
    parser.add_argument("--training-stage", default="base_pretrain")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scan = scan_prepack_sessions(args.sessions_root)
    _fail_on_completed_invalid(scan.invalid_sessions)
    existing = scan.total_valid_sequences
    if target_is_complete(existing, args.target_total_sequences):
        print("TARGET COMPLETE")
        return 0

    remaining = int(args.target_total_sequences) - existing
    session_target = (
        min(int(args.session_target_sequences), remaining)
        if args.session_target_sequences is not None
        else remaining
    )
    session_id = args.session_id or next_session_id(args.sessions_root)
    session_dir = Path(args.sessions_root) / session_id
    if args.dry_run:
        summary = {
            "dry_run": True,
            "session_id": session_id,
            "output_dir": str(session_dir),
            "existing_valid_sequences_before": existing,
            "remaining_sequences_before": remaining,
            "session_target_sequences": session_target,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    session_dir = ensure_session_dir_available(args.sessions_root, session_id)
    started_at = utc_now_iso()
    info: dict[str, Any] = {
        "session_id": session_id,
        "started_at": started_at,
        "completed_at": None,
        "status": "running",
        "config": str(args.config),
        "datasets_config": str(args.datasets_config),
        "synthetic_input": args.synthetic_input,
        "output_dir": str(session_dir),
        "target_total_sequences": int(args.target_total_sequences),
        "existing_valid_sequences_before": existing,
        "remaining_sequences_before": remaining,
        "session_target_sequences": session_target,
        "max_session_hours": float(args.max_session_hours),
        "training_stage": args.training_stage,
        "final_session_sequences": 0,
        "total_valid_sequences_after": existing,
    }
    _write_session_info(session_dir, info)
    try:
        result = pack_tpu_dataset(
            config_path=args.config,
            datasets_config_path=args.datasets_config,
            synthetic_input=args.synthetic_input,
            output_dir=session_dir,
            target_sequences=session_target,
            training_stage=args.training_stage,
            stop_after_seconds=_effective_stop_after_seconds(args.max_session_hours),
        )
        final_sequences = int(result.manifest.get("total_sequences", 0))
        expected_total_after = existing + final_sequences
        status = _status_from_result(
            stop_reason=result.stop_reason,
            total_after=expected_total_after,
            target_total=int(args.target_total_sequences),
            final_sequences=final_sequences,
        )
        info.update(
            {
                "completed_at": utc_now_iso(),
                "status": status,
                "final_session_sequences": final_sequences,
                "total_valid_sequences_after": expected_total_after,
                "packer_stop_reason": result.stop_reason,
            }
        )
        _write_session_info(session_dir, info)
        post_scan = scan_prepack_sessions(args.sessions_root)
        total_after = post_scan.total_valid_sequences
        current_valid = any(
            session.session_id == session_id for session in post_scan.valid_sessions
        )
        info["total_valid_sequences_after"] = total_after
        _write_session_info(session_dir, info)
        if final_sequences > 0 and not current_valid:
            raise RuntimeError(f"session output failed verification: {session_dir}")
        print(json.dumps(info, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        info.update(
            {
                "completed_at": utc_now_iso(),
                "status": "failed",
                "error": str(exc),
            }
        )
        _write_session_info(session_dir, info)
        raise


def _effective_stop_after_seconds(max_session_hours: float) -> float:
    if max_session_hours <= 0:
        raise ValueError("max_session_hours must be positive")
    return max(0.0, float(max_session_hours) * 3600.0 - SAFETY_BUFFER_SECONDS)


def _status_from_result(
    *,
    stop_reason: str,
    total_after: int,
    target_total: int,
    final_sequences: int,
) -> str:
    if target_is_complete(total_after, target_total):
        return "target_complete"
    if final_sequences <= 0:
        return "no_sequences_written"
    if stop_reason == "target_sequences":
        return "completed"
    if stop_reason == "time_guard":
        return "completed_time_guard"
    if stop_reason == "data_exhausted":
        return "completed_data_exhausted"
    return "completed"


def _fail_on_completed_invalid(invalid_sessions: list[Any]) -> None:
    completed_invalid = [session for session in invalid_sessions if session.completed]
    if not completed_invalid:
        return
    details = "; ".join(
        f"{session.session_id}: {session.reason}" for session in completed_invalid
    )
    raise RuntimeError(f"completed prepack session validation failed: {details}")


def _write_session_info(session_dir: Path, info: dict[str, Any]) -> None:
    write_json(session_dir / SESSION_INFO_FILENAME, info)


if __name__ == "__main__":
    raise SystemExit(main())
