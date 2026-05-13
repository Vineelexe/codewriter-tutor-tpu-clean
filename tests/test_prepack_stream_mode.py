from __future__ import annotations

from itertools import islice

from cw360.data.modes import DataPipelineMode
from cw360.data.pipeline import DataPipeline, PipelineOptions


class FIMTokenizer:
    eos_token = "<eos>"
    eos_token_id = 0
    all_special_tokens = ["<fim_prefix>", "<fim_suffix>", "<fim_middle>"]
    special_tokens_map = {
        "additional_special_tokens": ["<fim_prefix>", "<fim_suffix>", "<fim_middle>"]
    }

    def get_vocab(self) -> dict[str, int]:
        return {"<fim_prefix>": 1, "<fim_suffix>": 2, "<fim_middle>": 3}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [ord(char) for char in text]


def test_prepack_stream_returns_formatted_examples_with_mixture_stats() -> None:
    pipeline = DataPipeline(
        {
            "stack_v2_python": [
                {
                    "content": (
                        "def add(a, b):\n"
                        "    total = a + b\n"
                        "    return total\n"
                    )
                }
            ],
            "codesearchnet_python": [
                {
                    "code": "def clean(value):\n    return value.strip()\n",
                    "docstring": "Clean a string value for later comparison.",
                }
            ],
        },
        PipelineOptions(
            mode=DataPipelineMode.PREPACK_STREAM,
            training_stage="fim_train",
            seed=2,
        ),
        tokenizer=FIMTokenizer(),
        weights={"stack_v2_python": 0.5, "codesearchnet_python": 0.5},
    )

    examples = list(islice(pipeline.prepack_stream(), 2))
    stats = pipeline.mixture_stats()

    assert len(examples) == 2
    assert all(example.training_stage == "fim_train" for example in examples)
    assert all(example.quality_tier in {"A", "B"} for example in examples)
    assert any("<fim_prefix>" in example.text for example in examples)
    assert all(example.metadata["loss_weight"] == example.loss_weight for example in examples)
    assert stats is not None
    assert stats.total_emitted == 2


def test_prepack_instruction_stage_formats_prompt_response_when_available() -> None:
    pipeline = DataPipeline(
        {
            "synthetic": [
                {
                    "text": "placeholder",
                    "metadata": {
                        "prompt": "Write a Python add function.",
                        "response": "def add(a, b):\n    return a + b\n",
                        "task_type": "code_writing",
                    },
                    "source": "synthetic",
                    "example_type": "synthetic_instruction",
                }
            ]
        },
        PipelineOptions(
            mode=DataPipelineMode.PREPACK_STREAM,
            training_stage="instruction_tune",
        ),
        weights={"synthetic": 1.0},
    )

    example = next(pipeline.prepack_stream())

    assert example.example_type == "instruction_code_writing"
    assert example.text.startswith("User:\nWrite a Python add function.")
    assert "Assistant:\ndef add" in example.text
