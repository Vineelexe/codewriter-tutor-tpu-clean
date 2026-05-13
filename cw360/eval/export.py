from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from cw360.eval.scoring_schema import SCORING_SCHEMA


def write_jsonl(records: Sequence[dict[str, Any]], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return output_path


def write_markdown_report(records: Sequence[dict[str, Any]], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_name = records[0]["checkpoint_name"] if records else "unknown"
    output_lines = [
        f"# Evaluation Report: {checkpoint_name}",
        "",
        "## Run Metadata",
        "",
    ]
    if records:
        first = records[0]
        output_lines.extend(
            [
                f"- Date: {first['date']}",
                f"- Step: {first['step']}",
                f"- Training stage: {first['training_stage']}",
                f"- Model size label: {first['model_size_label']}",
                f"- Temperature: {first['temperature']}",
                f"- Max new tokens: {first['max_new_tokens']}",
                f"- Data snapshot: `{json.dumps(first['data_snapshot'], sort_keys=True)}`",
                f"- Notes: {first['notes']}",
                "",
            ]
        )

    output_lines.extend(
        [
            "## Scoring Schema",
            "",
        ]
    )
    for criterion in SCORING_SCHEMA:
        output_lines.append(
            f"- {criterion.name}: {criterion.min_score}-{criterion.max_score}. "
            f"{criterion.description}"
        )

    output_lines.extend(["", "## Outputs", ""])
    for record in records:
        output_lines.extend(
            [
                f"### {record['prompt_id']}",
                "",
                f"Category: `{record['category']}`",
                "",
                "**Prompt**",
                "",
                "```text",
                str(record["prompt"]),
                "```",
                "",
                "**Output**",
                "",
                "```text",
                str(record["output"]),
                "```",
                "",
            ]
        )

    output_path.write_text("\n".join(output_lines), encoding="utf-8", newline="\n")
    return output_path
