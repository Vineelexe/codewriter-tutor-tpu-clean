from __future__ import annotations

import argparse

from cw360.teacher.engine import (
    TeacherGenerationEngine,
    load_teacher_engine_config,
    summarize_result,
)
from cw360.teacher.providers.mock import MockTeacherClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate offline parent-LLM teacher work.")
    parser.add_argument("--config", default="configs/teacher_generation.yaml")
    args = parser.parse_args()

    config = load_teacher_engine_config(args.config)
    engine = TeacherGenerationEngine(
        config,
        client=MockTeacherClient(config.model),
        local_only=True,
    )
    print(summarize_result(engine.estimate()))


if __name__ == "__main__":
    main()
