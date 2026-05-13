from __future__ import annotations

from typing import Any

from cw360.data.schemas import TrainingExample
from cw360.synthetic.schemas import StructuredSyntheticRecord


def record_to_training_text(record: StructuredSyntheticRecord) -> str:
    return (
        f"User:\n{record.user_prompt.strip()}\n\n"
        f"Assistant:\n{record.assistant_response.strip()}\n"
    )


def record_to_training_example(record: StructuredSyntheticRecord) -> TrainingExample:
    metadata = record_metadata(record)
    return TrainingExample(
        text=record_to_training_text(record),
        source="structured_synthetic_instructions",
        example_type=f"instruction_{record.task_type}",
        metadata=metadata,
        quality_tier="A" if record.loss_weight >= 1.0 else "B",
        loss_weight=record.loss_weight,
        training_stage=record.train_stage,
    )


def record_to_jsonl_row(record: StructuredSyntheticRecord) -> dict[str, Any]:
    example = record_to_training_example(record)
    return {
        "text": example.text,
        "source": example.source,
        "example_type": example.example_type,
        "quality_tier": example.quality_tier,
        "loss_weight": example.loss_weight,
        "training_stage": example.training_stage,
        "metadata": dict(example.metadata),
    }


def record_metadata(record: StructuredSyntheticRecord) -> dict[str, Any]:
    metadata = dict(record.metadata)
    metadata.update(
        {
            "synthetic_id": record.synthetic_id,
            "source_candidate_id": record.source_candidate_id,
            "candidate_id": record.candidate_id,
            "source": record.source,
            "record_type": record.record_type,
            "task_type": record.task_type,
            "difficulty": record.difficulty,
            "skills": list(record.skills),
            "train_stage": record.train_stage,
            "loss_weight": record.loss_weight,
            "teacher_provider": record.teacher_provider,
            "teacher_model": record.teacher_model,
            "generation_time": record.generation_time,
            "input_hash": record.input_hash,
            "output_hash": record.output_hash,
            "prompt": record.user_prompt,
            "response": record.assistant_response,
            "prepack_ready": True,
            "preformatted_text": True,
        }
    )
    for key in ("code_before", "code_after", "explanation", "tests", "quality_notes"):
        value = getattr(record, key)
        if value is not None:
            metadata[key] = value
    return metadata
