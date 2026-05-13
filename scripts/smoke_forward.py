from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from cw360.config import load_model_config  # noqa: E402
from cw360.model.lm import CodeWriterTutorLM  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny CPU LM forward smoke test.")
    parser.add_argument("--config", type=Path, required=True, help="Path to model YAML config")
    args = parser.parse_args()

    config = load_model_config(args.config)
    if config.size_label != "tiny":
        raise SystemExit("smoke_forward.py is intended for tiny CPU configs only")

    model = CodeWriterTutorLM(config)
    model.eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 8), dtype=torch.long)
    output = model(input_ids, labels=input_ids)
    generated = model.generate(input_ids[:1, :4], max_new_tokens=2, use_cache=True)
    payload = {
        "logits_shape": list(output.logits.shape),
        "loss": float(output.loss.item()) if output.loss is not None else None,
        "generated_shape": list(generated.shape),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
