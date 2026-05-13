from __future__ import annotations

import argparse

from cw360.teacher.engine import (
    TeacherGenerationEngine,
    load_teacher_engine_config,
    override_config,
    summarize_result,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline parent-LLM data structuring.")
    parser.add_argument("--config", default="configs/teacher_generation.yaml")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()

    config = override_config(
        load_teacher_engine_config(args.config),
        provider=args.provider,
        max_requests=args.max_requests,
        dry_run=True if args.dry_run else None,
    )
    engine = TeacherGenerationEngine(config, local_only=args.local_only)
    print(summarize_result(engine.run()))


if __name__ == "__main__":
    main()
