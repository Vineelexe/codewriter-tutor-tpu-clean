from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cw360.prepack.session_runner import scan_prepack_sessions  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify and summarize prepack sessions.")
    parser.add_argument("--sessions-root", required=True)
    parser.add_argument("--target-total-sequences", type=int)
    args = parser.parse_args()

    scan = scan_prepack_sessions(args.sessions_root)
    print(f"total sessions: {scan.total_sessions}")
    print(f"completed sessions: {len(scan.valid_sessions)}")
    print(f"ignored sessions: {len(scan.ignored_sessions)}")
    print(f"invalid sessions: {len(scan.invalid_sessions)}")
    print(f"total shards: {scan.total_valid_shards}")
    print(f"total sequences: {scan.total_valid_sequences}")
    print(f"total training tokens: {scan.total_valid_training_tokens}")
    if args.target_total_sequences:
        percent = (
            scan.total_valid_sequences / float(args.target_total_sequences)
            if args.target_total_sequences > 0
            else 0.0
        )
        print(f"completion: {percent:.2%}")
    for invalid in scan.invalid_sessions:
        print(
            f"invalid {invalid.session_id}: completed={invalid.completed} "
            f"reason={invalid.reason}",
            file=sys.stderr,
        )
    for ignored in scan.ignored_sessions:
        print(f"ignored {ignored.session_id}: reason={ignored.reason}", file=sys.stderr)
    return 1 if any(session.completed for session in scan.invalid_sessions) else 0


if __name__ == "__main__":
    raise SystemExit(main())
