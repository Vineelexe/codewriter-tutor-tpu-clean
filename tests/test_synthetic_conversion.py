from __future__ import annotations

from cw360.data.modes import DataPipelineMode
from cw360.data.pipeline import DataPipeline, PipelineOptions
from cw360.synthetic.convert import record_to_jsonl_row, record_to_training_example
from cw360.synthetic.schemas import StructuredSyntheticRecord
from tests.test_synthetic_schema import sample_payload


def test_conversion_preserves_prepack_metadata_in_training_example() -> None:
    record = StructuredSyntheticRecord.from_dict(sample_payload())

    example = record_to_training_example(record)

    assert example.text.startswith("User:\nDebug this Python")
    assert example.example_type == "instruction_debugging"
    assert example.training_stage == "instruction_tune"
    assert example.loss_weight == 1.0
    assert example.metadata["synthetic_id"] == "synth-1"
    assert example.metadata["source_candidate_id"] == "candidate-1"
    assert example.metadata["skills"] == ["python", "debugging", "pytest"]
    assert example.metadata["prepack_ready"] is True


def test_converted_jsonl_row_is_consumable_by_prepack_stream() -> None:
    record = StructuredSyntheticRecord.from_dict(sample_payload())
    row = record_to_jsonl_row(record)
    pipeline = DataPipeline(
        {"synthetic": [row]},
        PipelineOptions(
            mode=DataPipelineMode.PREPACK_STREAM,
            training_stage="base_pretrain",
        ),
        weights={"synthetic": 1.0},
    )

    example = next(pipeline.prepack_stream())

    assert example.training_stage == "instruction_tune"
    assert example.text.startswith("User:\nDebug this Python")
    assert example.metadata["prepack_ready"] is True
    assert example.metadata["task_type"] == "debugging"
