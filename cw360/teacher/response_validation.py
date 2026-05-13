from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from cw360.teacher.structured_record import (
    DIFFICULTIES,
    RECORD_TYPES,
    TASK_TYPES,
    VALID_LOSS_WEIGHTS,
    StructuredTrainingRecord,
)

REQUIRED_FIELDS = frozenset(
    {
        "synthetic_id",
        "source_candidate_id",
        "source",
        "record_type",
        "task_type",
        "difficulty",
        "skills",
        "user_prompt",
        "assistant_response",
        "train_stage",
        "loss_weight",
        "teacher_provider",
        "teacher_model",
        "generation_time",
        "input_hash",
        "output_hash",
        "metadata",
    }
)

_GENERIC_FLUFF = (
    "as an ai language model",
    "i hope this helps",
    "here is some information",
    "it depends on your needs",
)
_PROVIDER_ERROR = (
    "api error",
    "rate limit",
    "unauthorized",
    "forbidden",
    "invalid api key",
    "service unavailable",
)
_PYTHON_MARKERS = (
    "python",
    "def ",
    "class ",
    "import ",
    "pytest",
    "traceback",
    "exception",
    "error",
    "list",
    "dict",
    "json",
    "csv",
)
_UNSAFE_PATTERNS = (
    r"os\.system\([^)]*(rm\s+-rf|del\s+/s|format\s+c:)",
    r"subprocess\.[a-z_]+\([^)]*(rm\s+-rf|del\s+/s|format\s+c:)",
    r"shutil\.rmtree\([\"']/(?:[\"']|\s*,)",
    r"\beval\s*\(\s*input\s*\(",
    r"\bexec\s*\(\s*input\s*\(",
)


@dataclass(slots=True)
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    record: StructuredTrainingRecord | None = None


def parse_json_response(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON response: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("JSON response must be an object")
    return parsed


def validate_response_payload(
    payload: dict[str, Any],
    *,
    source_prompt: str | None = None,
    min_response_chars: int = 24,
    max_response_chars: int = 12000,
) -> ValidationResult:
    errors: list[str] = []
    missing = sorted(REQUIRED_FIELDS - set(payload))
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")

    if payload.get("record_type") not in RECORD_TYPES:
        errors.append("invalid record_type")
    if payload.get("task_type") not in TASK_TYPES:
        errors.append("invalid task_type")
    if payload.get("difficulty") not in DIFFICULTIES:
        errors.append("invalid difficulty")
    if payload.get("train_stage") not in {
        "base_pretrain",
        "fim_train",
        "instruction_tune",
        "eval_only",
    }:
        errors.append("invalid train_stage")

    loss_weight = payload.get("loss_weight")
    if not isinstance(loss_weight, int | float) or not math.isfinite(float(loss_weight)):
        errors.append("invalid loss_weight")
    elif float(loss_weight) not in VALID_LOSS_WEIGHTS:
        errors.append("invalid loss_weight")

    skills = payload.get("skills")
    if not isinstance(skills, list) or not skills or not all(isinstance(s, str) for s in skills):
        errors.append("missing skills")

    user_prompt = str(payload.get("user_prompt") or "").strip()
    assistant_response = str(payload.get("assistant_response") or "").strip()
    if not user_prompt:
        errors.append("empty user_prompt")
    if not assistant_response:
        errors.append("empty assistant_response")
    if len(assistant_response) < min_response_chars:
        errors.append("assistant_response too short")
    if len(assistant_response) > max_response_chars:
        errors.append("assistant_response too long")
    if source_prompt and _too_similar(source_prompt, assistant_response):
        errors.append("assistant_response nearly identical to prompt")
    if _contains_any(assistant_response, _GENERIC_FLUFF):
        errors.append("generic fluff")
    if _contains_any(assistant_response, _PROVIDER_ERROR):
        errors.append("provider/API error text")
    combined = f"{user_prompt}\n{assistant_response}\n{' '.join(skills or [])}"
    if not _contains_any(combined, _PYTHON_MARKERS):
        errors.append("not Python-related")
    has_unsafe_code = any(
        re.search(pattern, assistant_response, flags=re.IGNORECASE)
        for pattern in _UNSAFE_PATTERNS
    )
    if has_unsafe_code:
        errors.append("unsafe/malicious code")

    if errors:
        return ValidationResult(ok=False, errors=errors)
    try:
        record = StructuredTrainingRecord.from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        return ValidationResult(ok=False, errors=[str(exc)])
    return ValidationResult(ok=True, record=record)


def validate_json_response(
    text: str,
    *,
    source_prompt: str | None = None,
) -> ValidationResult:
    try:
        payload = parse_json_response(text)
    except ValueError as exc:
        return ValidationResult(ok=False, errors=[str(exc)])
    return validate_response_payload(payload, source_prompt=source_prompt)


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(needle in lower for needle in needles)


def _too_similar(left: str, right: str) -> bool:
    clean_left = _normalize_for_similarity(left)
    clean_right = _normalize_for_similarity(right)
    if not clean_left or not clean_right:
        return False
    return SequenceMatcher(a=clean_left, b=clean_right).ratio() >= 0.88


def _normalize_for_similarity(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9_]+", " ", value.lower())).strip()
