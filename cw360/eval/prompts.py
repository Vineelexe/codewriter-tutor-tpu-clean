from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class EvalPrompt:
    prompt_id: str
    prompt: str
    category: str
    notes: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source: Path, line_number: int) -> EvalPrompt:
        try:
            prompt_id = str(data["prompt_id"])
            prompt = str(data["prompt"])
            category = str(data["category"])
        except KeyError as exc:
            raise ValueError(
                f"eval prompt {source}:{line_number} is missing required field {exc.args[0]!r}"
            ) from exc
        if not prompt_id:
            raise ValueError(f"eval prompt {source}:{line_number} has empty prompt_id")
        if not prompt:
            raise ValueError(f"eval prompt {source}:{line_number} has empty prompt")
        return cls(
            prompt_id=prompt_id,
            prompt=prompt,
            category=category,
            notes=None if data.get("notes") is None else str(data["notes"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "prompt": self.prompt,
            "category": self.category,
            "notes": self.notes,
        }


def load_eval_prompts(paths: Iterable[str | Path]) -> list[EvalPrompt]:
    prompts: list[EvalPrompt] = []
    seen_ids: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    loaded = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON in eval prompt {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(loaded, dict):
                    raise ValueError(f"eval prompt {path}:{line_number} must be a JSON object")
                prompt = EvalPrompt.from_dict(loaded, source=path, line_number=line_number)
                if prompt.prompt_id in seen_ids:
                    raise ValueError(f"duplicate eval prompt_id {prompt.prompt_id!r}")
                seen_ids.add(prompt.prompt_id)
                prompts.append(prompt)
    if not prompts:
        raise ValueError("at least one eval prompt is required")
    return prompts
