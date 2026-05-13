from __future__ import annotations

import re
from typing import Any

from cw360.constants import VALID_TRAINING_STAGES
from cw360.teacher.candidate import TeacherCandidate
from cw360.teacher.structured_record import VALID_LOSS_WEIGHTS

_SKILL_PATTERNS: tuple[tuple[str, str], ...] = (
    ("loops", r"\b(for|while)\b"),
    ("lists", r"\b(list|append|extend|\[[^\]]*\])\b"),
    ("dicts", r"\b(dict|\.get\(|keys\(|values\(|\{[^{}:]+:)\b"),
    ("files", r"\b(open\(|Path\(|read_text|write_text|with\s+open)\b"),
    ("CSV", r"\b(csv|DictReader|DictWriter)\b"),
    ("JSON", r"\b(json|loads|dumps|load\(|dump\()\b"),
    ("classes", r"\bclass\s+\w+|self\."),
    ("errors", r"\b(Exception|Error|Traceback|try:|except)\b"),
    ("recursion", r"\brecurs|return\s+\w+\("),
    ("tests", r"\bpytest|unittest|assert\b"),
)


def infer_skills(text: str, task_type: str | None = None) -> list[str]:
    found: list[str] = []
    for skill, pattern in _SKILL_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE):
            found.append(skill)
    if task_type == "tests" and "tests" not in found:
        found.append("tests")
    if task_type == "debugging" and "errors" not in found:
        found.append("errors")
    return found or ["python"]


def infer_difficulty(text: str, score: float | None = None) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    complexity_markers = sum(
        marker in text
        for marker in ("class ", "async ", "yield ", "recurs", "try:", "except", "pytest")
    )
    if len(lines) >= 35 or complexity_markers >= 3 or (score is not None and score >= 0.95):
        return "advanced"
    if len(lines) >= 12 or complexity_markers >= 1:
        return "intermediate"
    return "beginner"


def normalize_train_stage(stage: str | None, task_type: str) -> str:
    if stage in VALID_TRAINING_STAGES:
        return str(stage)
    if task_type in {"debugging", "explanation", "comments", "refactor", "tests", "code_writing"}:
        return "instruction_tune"
    return "base_pretrain"


def normalize_loss_weight(value: float | None, quality_tier: str | None = None) -> float:
    if value in VALID_LOSS_WEIGHTS:
        return float(value)
    if quality_tier == "A":
        return 1.5
    if quality_tier == "C":
        return 0.5
    return 1.0


def curriculum_metadata(candidate: TeacherCandidate) -> dict[str, Any]:
    text = candidate.text
    return {
        "difficulty": infer_difficulty(text, candidate.score),
        "skills": infer_skills(text, candidate.task_type),
        "train_stage": normalize_train_stage(candidate.proposed_train_stage, candidate.task_type),
        "loss_weight": normalize_loss_weight(
            candidate.proposed_loss_weight, candidate.quality_tier
        ),
        "quality_tier": candidate.quality_tier,
        "candidate_score": candidate.score,
        "reason_selected": candidate.reason_selected,
    }
