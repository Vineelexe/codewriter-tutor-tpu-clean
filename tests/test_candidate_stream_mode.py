from __future__ import annotations

from itertools import islice

from cw360.data.modes import DataPipelineMode
from cw360.data.pipeline import DataPipeline, PipelineOptions


def test_candidate_stream_returns_filtered_raw_training_examples_with_metadata() -> None:
    pipeline = DataPipeline(
        {
            "stack_v2_python": [
                {"content": "def ok(value):\n    return value + 1\n", "path": "ok.py"},
                {"content": "def broken(:\n    pass", "path": "bad.py"},
            ],
            "debugbench": [
                {
                    "error_message": "ValueError",
                    "buggy_code": "int('x')",
                    "fixed_code": "0",
                }
            ],
        },
        PipelineOptions(
            mode=DataPipelineMode.CANDIDATE_STREAM,
            training_stage="base_pretrain",
            seed=11,
        ),
        weights={"stack_v2_python": 0.5, "debugbench": 0.5},
    )

    examples = list(islice(pipeline.candidate_stream(), 2))

    assert len(examples) == 2
    assert all("def ok" in example.text or "buggy_code" in example.text for example in examples)
    assert all(example.metadata["source"] == example.source for example in examples)
    assert all(
        example.metadata["estimated_length_chars"] == len(example.text) for example in examples
    )
    assert all("stable_input_hash" in example.metadata for example in examples)
    assert pipeline.mixture_stats() is not None


def test_candidate_stream_preserves_original_safe_dataset_fields() -> None:
    pipeline = DataPipeline(
        {
            "stack_v2_python": [
                {
                    "content": "def f():\n    return 1\n",
                    "path": "pkg/f.py",
                    "repo": "example/repo",
                    "secret": "do-not-copy",
                }
            ]
        },
        PipelineOptions(mode=DataPipelineMode.CANDIDATE_STREAM),
        weights={"stack_v2_python": 1.0},
    )

    example = next(pipeline.candidate_stream())

    assert example.metadata["original_fields"]["repo"] == "example/repo"
    assert "secret" not in example.metadata["original_fields"]
    assert "content" not in example.metadata["original_fields"]
