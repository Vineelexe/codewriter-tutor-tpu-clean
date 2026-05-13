from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cw360.prepack.manifest import read_json
from cw360.prepack.verify import VerificationReport, verify_prepacked_dataset

SESSION_INFO_FILENAME = "session_info.json"
_SESSION_RE = re.compile(r"^session_(\d+)$")
_COMPLETED_STATUSES = frozenset(
    {
        "completed",
        "target_complete",
        "completed_time_guard",
        "completed_data_exhausted",
    }
)
_NON_COMPLETED_STATUSES = frozenset(
    {"running", "failed", "partial", "dry_run", "no_sequences_written"}
)


@dataclass(frozen=True, slots=True)
class ValidPrepackSession:
    session_id: str
    path: Path
    manifest: dict[str, Any]
    report: VerificationReport
    session_info: dict[str, Any]


@dataclass(frozen=True, slots=True)
class InvalidPrepackSession:
    session_id: str
    path: Path
    reason: str
    completed: bool
    session_info: dict[str, Any]


@dataclass(frozen=True, slots=True)
class IgnoredPrepackSession:
    session_id: str
    path: Path
    reason: str
    session_info: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PrepackSessionScan:
    sessions_root: Path
    total_sessions: int
    valid_sessions: list[ValidPrepackSession]
    invalid_sessions: list[InvalidPrepackSession]
    ignored_sessions: list[IgnoredPrepackSession]

    @property
    def total_valid_sequences(self) -> int:
        return sum(session.report.total_sequences for session in self.valid_sessions)

    @property
    def total_valid_shards(self) -> int:
        return sum(session.report.shard_count for session in self.valid_sessions)

    @property
    def total_valid_training_tokens(self) -> int:
        return sum(session.report.total_tokens_for_training for session in self.valid_sessions)


def session_directories(sessions_root: str | Path) -> list[Path]:
    root = Path(sessions_root)
    if not root.exists():
        return []
    return sorted(path for path in root.glob("session_*") if path.is_dir())


def next_session_id(sessions_root: str | Path) -> str:
    max_index = 0
    for path in session_directories(sessions_root):
        match = _SESSION_RE.match(path.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return f"session_{max_index + 1:04d}"


def ensure_session_dir_available(sessions_root: str | Path, session_id: str) -> Path:
    root = Path(sessions_root)
    session_dir = root / session_id
    if session_dir.exists() and any(session_dir.iterdir()):
        raise FileExistsError(
            f"session directory is non-empty and will not be overwritten: {session_dir}"
        )
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def target_is_complete(existing_valid_sequences: int, target_total_sequences: int) -> bool:
    if target_total_sequences <= 0:
        raise ValueError("target_total_sequences must be positive")
    return int(existing_valid_sequences) >= int(target_total_sequences)


def scan_prepack_sessions(sessions_root: str | Path) -> PrepackSessionScan:
    root = Path(sessions_root)
    valid: list[ValidPrepackSession] = []
    invalid: list[InvalidPrepackSession] = []
    ignored: list[IgnoredPrepackSession] = []
    dirs = session_directories(root)
    for session_dir in dirs:
        session_info = load_session_info(session_dir)
        completed = _session_info_completed(session_info)
        manifest_path = session_dir / "manifest.json"
        if not manifest_path.exists():
            if not completed:
                ignored.append(
                    IgnoredPrepackSession(
                        session_id=session_dir.name,
                        path=session_dir,
                        reason="manifest.json is missing",
                        session_info=session_info,
                    )
                )
                continue
            invalid.append(
                InvalidPrepackSession(
                    session_id=session_dir.name,
                    path=session_dir,
                    reason="manifest.json is missing",
                    completed=True,
                    session_info=session_info,
                )
            )
            continue
        try:
            report = verify_prepacked_dataset(session_dir)
            manifest = read_json(manifest_path)
        except Exception as exc:
            if not completed:
                ignored.append(
                    IgnoredPrepackSession(
                        session_id=session_dir.name,
                        path=session_dir,
                        reason=str(exc),
                        session_info=session_info,
                    )
                )
                continue
            invalid.append(
                InvalidPrepackSession(
                    session_id=session_dir.name,
                    path=session_dir,
                    reason=str(exc),
                    completed=True,
                    session_info=session_info,
                )
            )
            continue
        if not completed:
            ignored.append(
                IgnoredPrepackSession(
                    session_id=session_dir.name,
                    path=session_dir,
                    reason=_ignored_status_reason(session_info),
                    session_info=session_info,
                )
            )
            continue
        valid.append(
            ValidPrepackSession(
                session_id=session_dir.name,
                path=session_dir,
                manifest=manifest,
                report=report,
                session_info=session_info,
            )
        )
    return PrepackSessionScan(
        sessions_root=root,
        total_sessions=len(dirs),
        valid_sessions=valid,
        invalid_sessions=invalid,
        ignored_sessions=ignored,
    )


def load_session_info(session_dir: str | Path) -> dict[str, Any]:
    path = Path(session_dir) / SESSION_INFO_FILENAME
    if not path.exists():
        return {}
    try:
        return read_json(path)
    except Exception:
        return {}


def _session_info_completed(session_info: dict[str, Any]) -> bool:
    status = str(session_info.get("status") or "")
    if status in _COMPLETED_STATUSES:
        return True
    if status in _NON_COMPLETED_STATUSES:
        return False
    return False


def _ignored_status_reason(session_info: dict[str, Any]) -> str:
    status = str(session_info.get("status") or "")
    if not status:
        return "session_info.json is missing or has no completed status"
    if status in _NON_COMPLETED_STATUSES:
        return f"session status is not completed: {status}"
    return f"session status is unknown: {status}"
