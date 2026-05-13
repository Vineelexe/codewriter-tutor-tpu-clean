from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cw360.config import load_config  # noqa: E402
from cw360.train.cpu_trainer import CPUTinyTrainer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local tiny CPU training.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir or config.get("checkpoint_dir", "outputs/tiny_train"))
    trainer = CPUTinyTrainer(
        config=config,
        output_dir=output_dir,
        resume_checkpoint=args.resume,
    )
    result = trainer.train()
    print(
        json.dumps(
            {
                "checkpoint": str(result.checkpoints[-1]),
                "final_step": result.state.step,
                "tokens_seen": result.state.tokens_seen,
                "sequences_seen": result.state.sequences_seen,
                "resumed_from": None if result.resumed_from is None else str(result.resumed_from),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
