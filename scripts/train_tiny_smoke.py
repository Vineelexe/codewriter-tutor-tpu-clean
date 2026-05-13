from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cw360.config import load_config  # noqa: E402
from cw360.train.cpu_trainer import CPUTinyTrainer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run save/resume tiny CPU training smoke.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir or _timestamped_output_dir("outputs/tiny_train_smoke"))
    first_config = dict(config)
    first_config["max_steps"] = 1
    first = CPUTinyTrainer(config=first_config, output_dir=output_dir)
    first_result = first.train()

    second_config = dict(config)
    second_config["max_steps"] = max(2, int(config.get("max_steps", 2)))
    second = CPUTinyTrainer(
        config=second_config,
        output_dir=output_dir,
        resume_checkpoint=first_result.checkpoints[-1],
    )
    second_result = second.train()
    print(
        json.dumps(
            {
                "saved": str(first_result.checkpoints[-1]),
                "resumed_from": str(first_result.checkpoints[-1]),
                "final_checkpoint": str(second_result.checkpoints[-1]),
                "final_step": second_result.state.step,
                "tokens_seen": second_result.state.tokens_seen,
                "sequences_seen": second_result.state.sequences_seen,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _timestamped_output_dir(base: str) -> str:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return str(Path(base) / stamp)


if __name__ == "__main__":
    main()
