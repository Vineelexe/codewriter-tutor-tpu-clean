from __future__ import annotations

import json
from typing import Any

from cw360.teacher.curriculum import (
    infer_difficulty,
    infer_skills,
    normalize_loss_weight,
    normalize_train_stage,
)
from cw360.teacher.providers.base import BaseTeacherClient, TeacherResponse
from cw360.teacher.structured_record import (
    make_synthetic_id,
    stable_hash,
    stable_json_hash,
    utc_now_iso,
)


class MockTeacherClient(BaseTeacherClient):
    provider_name = "mock"

    def __init__(self, model_name: str = "mock-teacher") -> None:
        self.model_name = model_name
        self.calls = 0

    def validate_environment(self) -> None:
        return None

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> TeacherResponse:
        self.calls += 1
        candidate = _extract_candidate(prompt)
        candidate_id = str(candidate.get("candidate_id", f"mock-{self.calls}"))
        task_type = str(candidate.get("task_type", "code_writing"))
        source = str(candidate.get("source", "mock"))
        text = str(candidate.get("selected_text") or candidate.get("raw_text") or "")
        user_prompt = _user_prompt(task_type, text)
        assistant_response = _assistant_response(task_type, text)
        request_hash = stable_hash(prompt)
        output_basis = {
            "candidate_id": candidate_id,
            "assistant_response": assistant_response,
            "provider": self.provider_name,
            "model": self.model_name,
        }
        output_hash = stable_json_hash(output_basis)
        payload: dict[str, Any] = {
            "synthetic_id": make_synthetic_id(candidate_id, request_hash, output_hash),
            "source_candidate_id": candidate_id,
            "source": source,
            "record_type": "structured",
            "task_type": task_type,
            "difficulty": infer_difficulty(text, float(candidate.get("score", 0.0))),
            "skills": infer_skills(text, task_type),
            "user_prompt": user_prompt,
            "assistant_response": assistant_response,
            "train_stage": normalize_train_stage(candidate.get("proposed_train_stage"), task_type),
            "loss_weight": normalize_loss_weight(
                float(candidate.get("proposed_loss_weight", 1.0)),
                str(candidate.get("quality_tier", "B")),
            ),
            "teacher_provider": self.provider_name,
            "teacher_model": self.model_name,
            "generation_time": utc_now_iso(),
            "input_hash": str(candidate.get("input_hash") or stable_hash(text)),
            "output_hash": output_hash,
            "metadata": {
                "mock": True,
                "source_score": candidate.get("score"),
                "quality_tier": candidate.get("quality_tier"),
            },
        }
        if task_type == "debugging":
            payload["code_before"] = text
            payload["explanation"] = "The issue is explained in Python terms with a safer fix."
        if task_type == "tests":
            payload["tests"] = "def test_example_behavior():\n    assert True\n"
        return TeacherResponse(
            text=json.dumps(payload, sort_keys=True),
            provider_name=self.provider_name,
            model_name=self.model_name,
            request_id=f"mock-{self.calls}",
            raw={"mock": True},
        )


def _extract_candidate(prompt: str) -> dict[str, Any]:
    marker = "CANDIDATE_JSON:"
    if marker not in prompt:
        return {}
    raw = prompt.split(marker, 1)[1].strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _user_prompt(task_type: str, text: str) -> str:
    preview = text.strip()[:1200]
    if task_type == "debugging":
        return f"Debug this Python example and explain the fix:\n\n{preview}"
    if task_type == "tests":
        return f"Write focused pytest tests for this Python behavior:\n\n{preview}"
    if task_type == "comments":
        return f"Add clear comments or docstrings to this Python code:\n\n{preview}"
    if task_type == "refactor":
        return f"Refactor this Python code while preserving behavior:\n\n{preview}"
    if task_type == "explanation":
        return f"Explain what this Python code does:\n\n{preview}"
    return f"Write a correct Python solution for this task:\n\n{preview}"


def _assistant_response(task_type: str, text: str) -> str:
    if task_type == "debugging":
        return (
            "The Python failure comes from an unchecked edge case. Add a guard before the "
            "operation, keep the return type stable, and cover the failing path with a test."
        )
    if task_type == "tests":
        return (
            "Use pytest to exercise the normal Python path and the edge case. Keep assertions "
            "focused on behavior instead of implementation details."
        )
    if task_type == "comments":
        return (
            "The Python code should explain intent at the function boundary and avoid comments "
            "that merely repeat each statement."
        )
    if task_type == "refactor":
        return (
            "A safer Python refactor keeps the public behavior unchanged, gives intermediate "
            "values clear names, and separates validation from transformation."
        )
    if task_type == "explanation":
        return (
            "This Python example processes input step by step, handles the core data structure, "
            "and returns a deterministic result for the caller."
        )
    return (
        "A correct Python solution should validate inputs, use straightforward control flow, "
        "and return a deterministic result that can be tested with pytest."
    )
