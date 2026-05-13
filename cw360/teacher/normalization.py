from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from cw360.teacher.curriculum import (
    infer_difficulty,
    infer_skills,
    normalize_loss_weight,
    normalize_train_stage,
)
from cw360.teacher.queue import TeacherRequest
from cw360.teacher.structured_record import (
    DIFFICULTIES,
    make_synthetic_id,
    stable_json_hash,
    utc_now_iso,
)

CONTENT_FIELDS = frozenset(
    {
        "difficulty",
        "skills",
        "user_prompt",
        "assistant_response",
        "code_before",
        "code_after",
        "explanation",
        "tests",
        "quality_notes",
    }
)
TRUSTED_FIELDS = frozenset(
    {
        "synthetic_id",
        "source_candidate_id",
        "source",
        "record_type",
        "task_type",
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
OPTIONAL_CONTENT_FIELDS = (
    "code_before",
    "code_after",
    "explanation",
    "tests",
    "quality_notes",
)
_SAFE_RAW_JSON_LIMIT = 4096
_SENSITIVE_KEY_PARTS = (
    "api_key",
    "authorization",
    "password",
    "secret",
    "token",
)


def normalize_teacher_payload(
    raw_payload: dict[str, Any],
    *,
    request: TeacherRequest,
    provider_name: str,
    model_name: str,
    response_request_id: str | None = None,
    response_raw: dict[str, Any] | None = None,
    record_type: str = "structured",
) -> dict[str, Any]:
    candidate = request.candidate
    task_type = candidate.task_type
    difficulty = _difficulty(raw_payload, candidate.text, candidate.score)
    skills = _sanitize_skills(raw_payload.get("skills"), candidate.text, task_type)

    content_payload: dict[str, Any] = {
        "difficulty": difficulty,
        "skills": skills,
        "user_prompt": _clean_text(raw_payload.get("user_prompt")),
        "assistant_response": _clean_text(raw_payload.get("assistant_response")),
    }
    for key in OPTIONAL_CONTENT_FIELDS:
        value = _clean_text(raw_payload.get(key))
        if value:
            content_payload[key] = value

    output_hash = stable_json_hash(content_payload)
    synthetic_id = make_synthetic_id(
        candidate.candidate_id,
        request.request_hash,
        output_hash,
    )
    blocked_fields = sorted(TRUSTED_FIELDS.intersection(raw_payload))
    metadata: dict[str, Any] = {
        "normalized_by_engine": True,
        "candidate_score": candidate.score,
        "quality_tier": candidate.quality_tier,
        "reason_selected": candidate.reason_selected,
        "request_hash": request.request_hash,
    }
    if response_request_id:
        metadata["teacher_response_request_id"] = response_request_id
    safe_raw = _safe_response_raw(response_raw)
    if safe_raw is not None:
        metadata["teacher_response_raw"] = safe_raw
    if blocked_fields:
        metadata["llm_supplied_blocked_fields"] = blocked_fields

    payload: dict[str, Any] = {
        **content_payload,
        "synthetic_id": synthetic_id,
        "source_candidate_id": candidate.candidate_id,
        "source": candidate.source,
        "record_type": record_type,
        "task_type": task_type,
        "train_stage": normalize_train_stage(
            candidate.proposed_train_stage,
            task_type,
        ),
        "loss_weight": normalize_loss_weight(
            candidate.proposed_loss_weight,
            candidate.quality_tier,
        ),
        "teacher_provider": str(provider_name or "unknown"),
        "teacher_model": str(model_name or "unknown"),
        "generation_time": utc_now_iso(),
        "input_hash": candidate.input_hash,
        "output_hash": output_hash,
        "metadata": metadata,
    }
    return payload


def _difficulty(payload: Mapping[str, Any], candidate_text: str, score: float) -> str:
    value = str(payload.get("difficulty") or "").strip().lower()
    if value in DIFFICULTIES:
        return value
    return infer_difficulty(candidate_text, score)


def _sanitize_skills(value: Any, candidate_text: str, task_type: str) -> list[str]:
    raw_values: list[Any]
    if isinstance(value, list):
        raw_values = value
    elif isinstance(value, str):
        raw_values = [part for part in value.replace(";", ",").split(",")]
    else:
        raw_values = []

    skills: list[str] = []
    seen: set[str] = set()
    for item in raw_values:
        skill = str(item).strip()
        if not skill:
            continue
        key = skill.lower()
        if key in seen:
            continue
        seen.add(key)
        skills.append(skill)
    return skills or infer_skills(candidate_text, task_type)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_response_raw(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if not value:
        return None
    scrubbed = _scrub_raw(value)
    encoded = json.dumps(scrubbed, sort_keys=True, ensure_ascii=True)
    if len(encoded) > _SAFE_RAW_JSON_LIMIT:
        return None
    return scrubbed if isinstance(scrubbed, dict) else None


def _scrub_raw(value: Any, *, depth: int = 0) -> Any:
    if depth >= 5:
        return "<omitted>"
    if isinstance(value, Mapping):
        scrubbed: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _is_sensitive_key(key_text):
                scrubbed[key_text] = "<redacted>"
            else:
                scrubbed[key_text] = _scrub_raw(item, depth=depth + 1)
        return scrubbed
    if isinstance(value, list):
        return [_scrub_raw(item, depth=depth + 1) for item in value[:20]]
    if isinstance(value, str):
        if len(value) > 500:
            return value[:500] + "...<truncated>"
        return value
    if value is None or isinstance(value, bool | int | float):
        return value
    return str(value)


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in _SENSITIVE_KEY_PARTS)
