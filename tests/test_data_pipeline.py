from __future__ import annotations

from itertools import islice

from cw360.data.pipeline import DataPipeline, PipelineOptions
from cw360.data.schemas import TrainingExample


class TinyTokenizer:
    eos_token = "<eos>"
    eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [ord(char) for char in text]


def _sources() -> dict[str, list[dict[str, str]]]:
    return {
        "stack_v2_python": [
            {"content": "def add(a, b):\n    return a + b\n", "path": "math.py"},
            {"content": "def sub(a, b):\n    return a - b\n", "path": "math.py"},
        ],
        "codesearchnet_python": [
            {
                "code": "def clean(value):\n    return value.strip()\n",
                "docstring": "Clean a string value.",
            }
        ],
        "debugbench": [
            {
                "error_message": "ZeroDivisionError",
                "buggy_code": "x = 1 / 0",
                "fixed_code": "x = None",
            }
        ],
        "synthetic": [
            {
                "text": "Write and explain a Python function.",
                "source": "synthetic",
                "example_type": "synthetic_instruction",
            }
        ],
        "fineweb_edu": [{"text": "Python algorithm lessons explain function complexity."}],
        "cosmopedia_cs": [{"text": "Python graph algorithms use queues and data structures."}],
    }


def test_pipeline_extracts_filters_enriches_tokenizes_packs_and_batches() -> None:
    tokenizer = TinyTokenizer()
    pipeline = DataPipeline(
        _sources(),
        PipelineOptions(max_seq_len=32, batch_size=2, seed=5),
        tokenizer=tokenizer,
    )

    examples = list(islice(pipeline.prepack_stream(), 4))

    assert examples
    assert all(example.quality_tier in {"A", "B"} for example in examples)
    assert all("stable_input_hash" in example.metadata for example in examples)
    assert all(example.metadata["estimated_tokens"] > 0 for example in examples)

    pipeline = DataPipeline(
        _sources(),
        PipelineOptions(max_seq_len=32, batch_size=2, seed=5),
        tokenizer=tokenizer,
    )
    batch = next(pipeline.batch_stream(pad_token_id=0))

    assert batch.input_ids.shape == (2, 32)
    assert batch.labels.shape == (2, 32)
    assert pipeline.mixture_stats() is not None


def test_pipeline_accepts_training_example_iterators_directly() -> None:
    pipeline = DataPipeline(
        {
            "synthetic": [
                TrainingExample(
                    text="def direct():\n    return 'ok'\n",
                    source="synthetic",
                    example_type="synthetic_instruction",
                )
            ]
        },
        PipelineOptions(training_stage="instruction_tune", seed=1),
        weights={"synthetic": 1.0},
    )

    example = next(pipeline.candidate_stream())

    assert example.source == "synthetic"
    assert example.metadata["source"] == "synthetic"
    assert example.training_stage == "instruction_tune"
