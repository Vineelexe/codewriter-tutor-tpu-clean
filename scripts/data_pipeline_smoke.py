from __future__ import annotations

import argparse
import json
import sys
from itertools import islice
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cw360.config import load_config  # noqa: E402
from cw360.data.modes import DataPipelineMode, parse_pipeline_mode  # noqa: E402
from cw360.data.pipeline import DataPipeline, PipelineOptions  # noqa: E402


class SmokeTokenizer:
    eos_token = "<eos>"
    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if add_special_tokens:
            raise ValueError("smoke tokenizer does not add special tokens")
        return [1 + (ord(char) % 32000) for char in text]


def synthetic_sources() -> dict[str, list[dict[str, Any]]]:
    base: dict[str, list[dict[str, Any]]] = {
        "stack_v2_python": _expand(
            [
            {
                "content": "def add(a, b):\n    return a + b\n",
                "path": "math_utils.py",
            },
            {
                "content": "class Counter:\n    def inc(self, x):\n        return x + 1\n",
                "path": "counter.py",
            },
            ],
            "content",
        ),
        "codesearchnet_python": _expand(
            [
            {
                "code": "def normalize(value):\n    return value.strip().lower()\n",
                "docstring": "Normalize a string value for comparison.",
            },
            {
                "code": "def area(radius):\n    return radius * radius\n",
                "docstring": "Compute an approximate circle area.",
            },
            ],
            "code",
        ),
        "debugbench": _expand(
            [
            {
                "error_message": "ZeroDivisionError: division by zero",
                "traceback": "Traceback (most recent call last): ...",
                "buggy_code": "def div(a, b):\n    return a / b\n",
                "fixed_code": "def div(a, b):\n    return None if b == 0 else a / b\n",
            },
            {
                "error": "ValueError: invalid literal for int()",
                "buggy_code": "value = int(raw)\n",
                "fixed_code": "value = int(raw) if raw.isdigit() else 0\n",
            },
            ],
            "buggy_code",
        ),
        "synthetic": _expand(
            [
            {
                "text": (
                    "User:\nWrite a Python clamp function.\n\nAssistant:\n"
                    "def clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n"
                ),
                "source": "synthetic",
                "example_type": "synthetic_instruction",
            },
            {
                "text": (
                    "Explain how to debug a Python KeyError with a minimal "
                    "reproducible example."
                ),
                "source": "synthetic",
                "example_type": "synthetic_instruction",
            },
            ],
            "text",
        ),
        "fineweb_edu": _expand(
            [
            {"text": "Python functions help explain algorithm complexity and data structures."}
            ],
            "text",
        ),
        "cosmopedia_cs": _expand(
            [
            {"text": "A graph algorithm can use Python queues for breadth first search."}
            ],
            "text",
        ),
    }
    return base


def _expand(rows: list[dict[str, Any]], text_key: str, *, copies: int = 24) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for index in range(copies):
        for row in rows:
            clone = dict(row)
            clone[text_key] = f"{row[text_key]}\n# smoke_variant_{index}\n"
            if "path" in clone:
                clone["path"] = f"variant_{index}_{clone['path']}"
            expanded.append(clone)
    return expanded


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 8 data pipeline smoke checks.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--num-examples", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)
    mode = parse_pipeline_mode(args.mode)
    tokenizer = SmokeTokenizer()
    options = PipelineOptions(
        mode=mode,
        training_stage=str(config.get("training_stage", "base_pretrain")),
        max_seq_len=int(config.get("max_seq_len", 512)),
        batch_size=int(config.get("micro_batch_size", 2)),
        seed=13,
    )
    pipeline = DataPipeline(synthetic_sources(), options, tokenizer=tokenizer)

    if mode is DataPipelineMode.CANDIDATE_STREAM:
        examples = list(islice(pipeline.candidate_stream(), args.num_examples))
        print_examples("candidate_stream", examples)
        print_stats(pipeline)
        return 0

    if mode is DataPipelineMode.PREPACK_STREAM:
        examples = list(islice(pipeline.prepack_stream(), args.num_examples))
        print_examples("prepack_stream", examples)
        print_stats(pipeline)
        return 0

    batches = list(
        islice(pipeline.batch_stream(pad_token_id=tokenizer.pad_token_id), args.num_batches)
    )
    for index, batch in enumerate(batches, start=1):
        print(
            json.dumps(
                {
                    "batch": index,
                    "input_ids_shape": list(batch.input_ids.shape),
                    "labels_shape": list(batch.labels.shape),
                    "attention_mask_shape": list(batch.attention_mask.shape),
                },
                sort_keys=True,
            )
        )
    print_stats(pipeline)
    return 0


def print_examples(label: str, examples: list[Any]) -> None:
    for index, example in enumerate(examples, start=1):
        print(
            json.dumps(
                {
                    "stream": label,
                    "index": index,
                    "source": example.source,
                    "example_type": example.example_type,
                    "quality_tier": example.quality_tier,
                    "training_stage": example.training_stage,
                    "loss_weight": example.loss_weight,
                    "text_preview": example.text[:100].replace("\n", "\\n"),
                    "metadata_keys": sorted(example.metadata)[:12],
                },
                sort_keys=True,
            )
        )


def print_stats(pipeline: DataPipeline) -> None:
    stats = pipeline.mixture_stats()
    if stats is None:
        print(json.dumps({"mixture_stats": None}, sort_keys=True))
        return
    print(json.dumps({"mixture_stats": stats.to_dict()}, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
