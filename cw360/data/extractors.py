from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cw360.data.schemas import TrainingExample


def _first_text(row: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _metadata(row: Mapping[str, Any], *, exclude: set[str]) -> dict[str, Any]:
    return {str(key): value for key, value in row.items() if key not in exclude}


def extract_stack_v2_python(row: Mapping[str, Any]) -> TrainingExample:
    text = _first_text(row, ("content", "text", "code"))
    return TrainingExample(
        text=text,
        source="stack_v2_python",
        example_type="python_code",
        metadata=_metadata(row, exclude={"content", "text", "code"}),
        training_stage="base_pretrain",
    )


def extract_codesearchnet_python(row: Mapping[str, Any]) -> TrainingExample:
    code = _first_text(row, ("code", "func_code", "function"))
    docstring = _first_text(row, ("docstring", "func_documentation_string", "documentation"))
    text = f'"""{docstring.strip()}"""\n{code.strip()}' if docstring else code
    metadata = _metadata(
        row,
        exclude={
            "code",
            "func_code",
            "function",
            "docstring",
            "func_documentation_string",
            "documentation",
        },
    )
    metadata["docstring"] = docstring
    return TrainingExample(
        text=text,
        source="codesearchnet_python",
        example_type="docstring_code_pair",
        metadata=metadata,
        training_stage="base_pretrain",
    )


def extract_debugbench(row: Mapping[str, Any]) -> TrainingExample:
    parts: list[str] = []
    for label, keys in (
        ("error", ("error", "error_message", "exception")),
        ("stack_trace", ("stack_trace", "traceback")),
        ("buggy_code", ("buggy_code", "buggy", "code")),
        ("fixed_code", ("fixed_code", "fix", "patch")),
    ):
        value = _first_text(row, keys)
        if value:
            parts.append(f"{label}:\n{value.strip()}")
    text = "\n\n".join(parts) if parts else _first_text(row, ("text", "prompt"))
    return TrainingExample(
        text=text,
        source="debugbench",
        example_type="debugging_case",
        metadata=_metadata(row, exclude=set()),
        training_stage="instruction_tune",
    )


def extract_fineweb_edu(row: Mapping[str, Any]) -> TrainingExample:
    return TrainingExample(
        text=_first_text(row, ("text", "content")),
        source="fineweb_edu",
        example_type="educational_text",
        metadata=_metadata(row, exclude={"text", "content"}),
        training_stage="base_pretrain",
    )


def extract_cosmopedia_cs(row: Mapping[str, Any]) -> TrainingExample:
    return TrainingExample(
        text=_first_text(row, ("text", "content", "article")),
        source="cosmopedia_cs",
        example_type="cs_algorithm_text",
        metadata=_metadata(row, exclude={"text", "content", "article"}),
        training_stage="base_pretrain",
    )


def extract_synthetic_jsonl(row: Mapping[str, Any]) -> TrainingExample:
    text = _first_text(row, ("text", "prompt", "completion"))
    if not text and isinstance(row.get("messages"), list):
        text = "\n".join(str(item) for item in row["messages"])
    source = str(row.get("source") or "synthetic")
    example_type = str(row.get("example_type") or row.get("type") or "synthetic_instruction")
    metadata = dict(row.get("metadata") or {})
    for key, value in row.items():
        if key not in {
            "text",
            "prompt",
            "completion",
            "messages",
            "source",
            "example_type",
            "type",
            "metadata",
        }:
            metadata[str(key)] = value
    return TrainingExample(
        text=text,
        source=source,
        example_type=example_type,
        metadata=metadata,
        quality_tier=row.get("quality_tier") if isinstance(row.get("quality_tier"), str) else None,
        loss_weight=float(row["loss_weight"])
        if isinstance(row.get("loss_weight"), int | float)
        else None,
        training_stage=str(row.get("training_stage"))
        if row.get("training_stage")
        else "instruction_tune",
    )


EXTRACTORS = {
    "stack_v2_python": extract_stack_v2_python,
    "codesearchnet_python": extract_codesearchnet_python,
    "debugbench": extract_debugbench,
    "fineweb_edu": extract_fineweb_edu,
    "cosmopedia_cs": extract_cosmopedia_cs,
    "synthetic": extract_synthetic_jsonl,
}


def extract_training_example(source_name: str, row: Mapping[str, Any]) -> TrainingExample:
    try:
        extractor = EXTRACTORS[source_name]
    except KeyError as exc:
        raise ValueError(f"unknown extractor/source: {source_name}") from exc
    return extractor(row)
