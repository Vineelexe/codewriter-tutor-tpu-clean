from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mine_teacher_candidates import mine_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview Phase 8.5 teacher candidates.")
    parser.add_argument("--config", default="configs/teacher_candidates.yaml")
    parser.add_argument("--source", default="fake")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--use-fake-data", action="store_true")
    args = parser.parse_args()

    candidates, _router = mine_candidates(
        config_path=args.config,
        source=args.source,
        limit=args.limit,
        max_candidates=args.limit,
        min_score=args.min_score,
        use_fake_data=args.use_fake_data,
    )
    for candidate in candidates[: args.limit]:
        preview = candidate.text.replace("\n", "\\n")[:120]
        print(
            f"candidate_id={candidate.candidate_id} "
            f"source={candidate.source} "
            f"task_type={candidate.task_type} "
            f"score={candidate.score:.3f} "
            f"quality_tier={candidate.quality_tier} "
            f"proposed_train_stage={candidate.proposed_train_stage} "
            f"proposed_loss_weight={candidate.proposed_loss_weight:.2f} "
            f"reason_selected={candidate.reason_selected} "
            f"text={preview}"
        )


if __name__ == "__main__":
    main()
