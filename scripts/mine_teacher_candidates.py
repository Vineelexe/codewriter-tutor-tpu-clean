from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cw360.data.schemas import ExtractedRecord, TrainingExample
from cw360.data.sources import load_dataset_source_configs
from cw360.data.streaming import iter_source_records
from cw360.teacher.candidate import TeacherCandidate
from cw360.teacher.candidate_router import CandidateRouter, TeacherCandidateConfig
from cw360.teacher.candidate_shards import CandidateShardWriter


def fake_source_rows(source: str = "fake") -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {
        "debugbench": [
            {
                "error_message": "ZeroDivisionError",
                "stack_trace": (
                    "Traceback (most recent call last):\n"
                    '  File "calc.py", line 2\n'
                    "ZeroDivisionError: division by zero"
                ),
                "buggy_code": "def average(values):\n    return sum(values) / len(values)\n",
                "fixed_code": (
                    "def average(values):\n"
                    "    if not values:\n"
                    "        return 0\n"
                    "    return sum(values) / len(values)\n"
                ),
            },
            {
                "error_message": "KeyError",
                "buggy_code": (
                    "def get_user_name(users, user_id):\n"
                    "    return users[user_id]['name']\n"
                ),
                "fixed_code": (
                    "def get_user_name(users, user_id):\n"
                    "    return users.get(user_id, {}).get('name', '')\n"
                ),
            },
        ],
        "codesearchnet_python": [
            {
                "docstring": "Normalize a user supplied label into a lowercase slug.",
                "code": (
                    "def slugify(label):\n"
                    '    """Normalize a user supplied label into a lowercase slug."""\n'
                    "    cleaned = label.strip().lower()\n"
                    "    return '-'.join(part for part in cleaned.split() if part)\n"
                ),
            },
            {
                "docstring": "Return the first matching item from a sequence.",
                "code": (
                    "def first_match(items, predicate):\n"
                    '    """Return the first item where predicate returns true."""\n'
                    "    for item in items:\n"
                    "        if predicate(item):\n"
                    "            return item\n"
                    "    return None\n"
                ),
            },
        ],
        "stack_v2_python": [
            {
                "content": (
                    "def clamp(value, lower, upper):\n"
                    "    if value < lower:\n"
                    "        return lower\n"
                    "    if value > upper:\n"
                    "        return upper\n"
                    "    return value\n"
                ),
                "path": "utils/math.py",
            },
            {
                "content": (
                    "def parse_int(value, default=0):\n"
                    "    try:\n"
                    "        return int(value)\n"
                    "    except (TypeError, ValueError):\n"
                    "        return default\n"
                ),
                "path": "utils/parse.py",
            },
            {
                "content": (
                    "def format_user(user):\n"
                    "    name=user.get('name','').strip()\n"
                    "    email=user.get('email','').strip().lower()\n"
                    "    if not name:\n"
                    "        name = email.split('@')[0]\n"
                    "    return f'{name} <{email}>'\n"
                ),
                "path": "utils/users.py",
            },
        ],
    }
    if source == "fake":
        return rows
    if source not in rows:
        raise ValueError(f"fake data does not include source {source}")
    return {source: rows[source]}


def iter_rows_for_source(
    source: str,
    *,
    config_path: Path,
    use_fake_data: bool,
) -> Iterable[TrainingExample | dict[str, Any]]:
    if use_fake_data or source == "fake":
        return fake_source_rows(source)
    config = TeacherCandidateConfig.from_yaml(config_path)
    dataset_config = Path(config.dataset_config)
    if not dataset_config.is_absolute():
        dataset_config = config_path.resolve().parent.parent / dataset_config
    configs = load_dataset_source_configs(dataset_config)
    if source not in configs:
        available = ", ".join(sorted(configs))
        raise ValueError(f"unknown source {source}; available sources: {available}")
    return {source: _examples_from_records(iter_source_records(configs[source]))}


def mine_candidates(
    *,
    config_path: str | Path,
    source: str,
    limit: int | None = None,
    max_candidates: int | None = None,
    min_score: float | None = None,
    use_fake_data: bool = False,
) -> tuple[list[TeacherCandidate], CandidateRouter]:
    path = Path(config_path)
    config = TeacherCandidateConfig.from_yaml(path)
    router = CandidateRouter(config, max_candidates=max_candidates, min_score_override=min_score)
    source_rows = iter_rows_for_source(source, config_path=path, use_fake_data=use_fake_data)
    candidates: list[TeacherCandidate] = []
    for source_name, rows in source_rows.items():  # type: ignore[union-attr]
        for candidate in router.route_many(rows, source=source_name, limit=limit):
            candidates.append(candidate)
    return candidates, router


def _examples_from_records(records: Iterable[ExtractedRecord]) -> Iterator[TrainingExample]:
    for record in records:
        if record.example is not None:
            yield record.example


def _print_preview(candidates: Iterable[TeacherCandidate]) -> None:
    for candidate in candidates:
        preview = candidate.text.replace("\n", "\\n")[:100]
        print(
            f"{candidate.candidate_id} source={candidate.source} task={candidate.task_type} "
            f"score={candidate.score:.3f} tier={candidate.quality_tier} "
            f"stage={candidate.proposed_train_stage} weight={candidate.proposed_loss_weight:.2f} "
            f"reason={candidate.reason_selected} text={preview}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine Phase 8.5 teacher candidates offline.")
    parser.add_argument("--config", default="configs/teacher_candidates.yaml")
    parser.add_argument("--source", default="fake")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/candidates")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--use-fake-data", action="store_true")
    args = parser.parse_args()

    candidates, router = mine_candidates(
        config_path=args.config,
        source=args.source,
        limit=args.limit,
        max_candidates=args.max_candidates,
        min_score=args.min_score,
        use_fake_data=args.use_fake_data,
    )
    if args.preview or args.dry_run:
        _print_preview(candidates)

    if not args.dry_run:
        writer = CandidateShardWriter(args.output_dir)
        manifest = writer.write_candidates(
            candidates,
            config={
                "config_path": args.config,
                "source": args.source,
                "min_score": args.min_score,
                "max_candidates": args.max_candidates,
            },
        )
        print(f"wrote {manifest.total_candidates} candidates to {args.output_dir}")
    else:
        print(
            f"dry_run selected={len(candidates)} rejected={router.rejected} "
            f"source_counts={dict(router.source_counts)} task_counts={dict(router.task_counts)}"
        )


if __name__ == "__main__":
    main()
