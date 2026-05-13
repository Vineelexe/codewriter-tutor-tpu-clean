from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any

from cw360.data.filters import (
    contains_dangerous_code,
    estimate_token_count,
    has_low_entropy,
    has_repeated_character_spam,
    has_repetition_spam,
    looks_binary_or_corrupt,
    looks_minified,
    mostly_license_boilerplate,
)

DEBUG_MARKERS = (
    "IndexError",
    "TypeError",
    "ValueError",
    "KeyError",
    "ZeroDivisionError",
    "Traceback",
    "stack trace",
    "exception",
    "buggy_code",
    "fixed_code",
)
EDGE_CASE_MARKERS = ("if ", "elif ", "else:", "try:", "except ", "raise ", "None", "empty")
DEPENDENCY_DUMP_RE = re.compile(r"(^|\n)\s*[A-Za-z0-9_.-]+(==|>=|<=|~=)\d", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class CandidateScore:
    accepted: bool
    score: float
    selected_text: str
    estimated_tokens: int
    quality_tier: str
    proposed_train_stage: str
    proposed_loss_weight: float
    reason: str
    task_preferences: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)


def score_teacher_candidate(
    *,
    source: str,
    text: str,
    metadata: dict[str, Any] | None = None,
    min_score: float = 0.70,
    min_input_tokens: int = 80,
    max_input_tokens: int = 1600,
) -> CandidateScore:
    metadata = dict(metadata or {})
    stripped = text.strip()
    if not stripped:
        return _reject("empty_text")
    rejection = _hard_rejection_reason(stripped)
    if rejection is not None:
        return _reject(rejection)

    selected, excerpted = _select_text(stripped, max_input_tokens=max_input_tokens)
    estimated_tokens = estimate_token_count(selected)
    features = _features(source, selected, metadata=metadata)
    score = _score_from_features(source, selected, features)
    preferences = _task_preferences(source, selected, features)
    reason_parts = _reason_parts(source, features, excerpted=excerpted)

    if estimated_tokens < min_input_tokens and not (
        features["debug_signal"] or features["function_or_class"]
    ):
        score -= 0.18
        reason_parts.append("too_short_without_python_signal")

    if estimate_token_count(stripped) > max_input_tokens and excerpted:
        metadata["excerpted_from_long_input"] = True
        metadata["original_estimated_tokens"] = estimate_token_count(stripped)
    elif estimate_token_count(stripped) > max_input_tokens:
        metadata["rescue_reason"] = "long_useful_input_needs_parent_llm_excerpt_review"

    score = max(0.0, min(1.0, round(score, 3)))
    quality_tier = _quality_tier(score, long_rescue=bool(metadata.get("rescue_reason")))
    accepted = score >= min_score or quality_tier == "RESCUE"
    if not accepted:
        return CandidateScore(
            accepted=False,
            score=score,
            selected_text=selected,
            estimated_tokens=estimated_tokens,
            quality_tier="C",
            proposed_train_stage="instruction_tune",
            proposed_loss_weight=0.0,
            reason="below_min_score:" + ",".join(reason_parts),
            task_preferences=preferences,
            metadata=metadata,
        )

    return CandidateScore(
        accepted=True,
        score=score,
        selected_text=selected,
        estimated_tokens=estimated_tokens,
        quality_tier=quality_tier,
        proposed_train_stage=_train_stage(preferences),
        proposed_loss_weight=_loss_weight(quality_tier, score),
        reason=",".join(reason_parts),
        task_preferences=preferences,
        metadata=metadata,
    )


def _reject(reason: str) -> CandidateScore:
    return CandidateScore(
        accepted=False,
        score=0.0,
        selected_text="",
        estimated_tokens=0,
        quality_tier="C",
        proposed_train_stage="instruction_tune",
        proposed_loss_weight=0.0,
        reason=reason,
    )


def _hard_rejection_reason(text: str) -> str | None:
    if looks_binary_or_corrupt(text):
        return "binary_or_corrupt_text"
    if contains_dangerous_code(text):
        return "unsafe_or_malware_like_code"
    if looks_minified(text):
        return "minified_or_obfuscated_text"
    if mostly_license_boilerplate(text):
        return "mostly_license_boilerplate"
    python_signal = "def " in text or "class " in text or "Traceback" in text
    if (
        not python_signal
        and (
            has_repeated_character_spam(text)
            or has_low_entropy(text)
            or has_repetition_spam(text)
        )
    ):
        return "corrupted_or_repetitive_text"
    dependency_lines = len(DEPENDENCY_DUMP_RE.findall(text))
    if dependency_lines >= 20:
        return "mostly_dependency_dump"
    return None


def _select_text(text: str, *, max_input_tokens: int) -> tuple[str, bool]:
    if estimate_token_count(text) <= max_input_tokens:
        return text, False

    max_chars = max_input_tokens * 4
    tree = _parse_python(text)
    if tree is not None:
        lines = text.splitlines()
        chunks: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
                continue
            start = max(0, int(node.lineno) - 3)
            end = min(len(lines), int(node.end_lineno or node.lineno) + 2)
            chunk = "\n".join(lines[start:end]).strip()
            if chunk and estimate_token_count(chunk) <= max_input_tokens:
                chunks.append(chunk)
        if chunks:
            chunks.sort(key=lambda item: (_function_quality_signal(item), -len(item)), reverse=True)
            return chunks[0], True

    lowered = text.lower()
    positions = [
        lowered.find(marker.lower())
        for marker in DEBUG_MARKERS + ("def ", "class ")
        if lowered.find(marker.lower()) >= 0
    ]
    center = min(positions) if positions else 0
    start = max(0, center - max_chars // 3)
    return text[start : start + max_chars].strip(), True


def _features(source: str, text: str, *, metadata: dict[str, Any]) -> dict[str, bool | int]:
    tree = _parse_python(text)
    docstring = str(metadata.get("docstring") or "")
    function_nodes: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    class_nodes: list[ast.ClassDef] = []
    if tree is not None:
        function_nodes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        class_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        docstring = docstring or ast.get_docstring(tree) or ""

    line_count = len([line for line in text.splitlines() if line.strip()])
    return {
        "valid_python": tree is not None,
        "function_or_class": bool(function_nodes or class_nodes),
        "compact_function": any(_node_line_count(node) <= 50 for node in function_nodes),
        "clean_explanation_size": 20 <= line_count <= 50,
        "docstring": bool(docstring.strip()) or '"""' in text or "'''" in text,
        "comments": "#" in text,
        "debug_signal": _has_debug_signal(source, text),
        "tests_signal": "assert " in text or "pytest" in text,
        "edge_cases": sum(marker in text for marker in EDGE_CASE_MARKERS),
        "messy": _messy_signal(text),
        "line_count": line_count,
    }


def _score_from_features(source: str, text: str, features: dict[str, bool | int]) -> float:
    score = 0.34
    if source == "debugbench" and features["debug_signal"]:
        score += 0.22
    if features["valid_python"]:
        score += 0.18
    if features["function_or_class"]:
        score += 0.18
    if features["compact_function"]:
        score += 0.10
    if features["docstring"]:
        score += 0.08
    if features["comments"]:
        score += 0.04
    if features["debug_signal"]:
        score += 0.20
    if features["tests_signal"]:
        score += 0.08
    if int(features["edge_cases"]) >= 2:
        score += 0.06
    if source == "debugbench":
        score += 0.12
    elif source == "codesearchnet_python":
        score += 0.09
    elif source == "stack_v2_python":
        score += 0.06
    if _trivial(text):
        score -= 0.10 if features["function_or_class"] else 0.25
    return score


def _task_preferences(
    source: str,
    text: str,
    features: dict[str, bool | int],
) -> tuple[str, ...]:
    if source == "debugbench" or features["debug_signal"]:
        return ("debugging", "tests", "explanation")
    if source == "codesearchnet_python" and features["docstring"]:
        return ("explanation", "comments", "tests")
    if features["messy"]:
        return ("refactor", "code_writing", "comments")
    if int(features["edge_cases"]) >= 2 or features["tests_signal"]:
        return ("tests", "code_writing", "explanation")
    if features["function_or_class"] and not features["comments"] and not features["docstring"]:
        return ("comments", "code_writing", "tests")
    if source == "stack_v2_python":
        return ("code_writing", "comments", "tests")
    return ("code_writing", "explanation", "comments")


def _reason_parts(
    source: str,
    features: dict[str, bool | int],
    *,
    excerpted: bool,
) -> list[str]:
    parts = [f"source={source}"]
    for key in (
        "function_or_class",
        "compact_function",
        "docstring",
        "debug_signal",
        "tests_signal",
        "messy",
    ):
        if features[key]:
            parts.append(str(key))
    if excerpted:
        parts.append("excerpted_long_useful_input")
    return parts


def _quality_tier(score: float, *, long_rescue: bool) -> str:
    if long_rescue:
        return "RESCUE"
    if score >= 0.85:
        return "A"
    if score >= 0.70:
        return "B"
    return "C"


def _loss_weight(quality_tier: str, score: float) -> float:
    if quality_tier == "A":
        return 1.0
    if quality_tier == "B":
        return round(max(0.45, min(0.85, score)), 2)
    if quality_tier == "RESCUE":
        return 0.2
    return 0.0


def _train_stage(task_preferences: tuple[str, ...]) -> str:
    if task_preferences and task_preferences[0] in {
        "debugging",
        "explanation",
        "comments",
        "refactor",
        "tests",
    }:
        return "instruction_tune"
    return "teacher_structuring"


def _parse_python(text: str) -> ast.AST | None:
    try:
        return ast.parse(text)
    except SyntaxError:
        return None


def _has_debug_signal(source: str, text: str) -> bool:
    lowered = text.lower()
    if source == "debugbench":
        return any(marker.lower() in lowered for marker in DEBUG_MARKERS)
    strong_context = any(
        marker in lowered
        for marker in (
            "traceback",
            "stack trace",
            "buggy_code",
            "fixed_code",
            "error:",
            "exception:",
        )
    )
    if strong_context:
        return any(marker.lower() in lowered for marker in DEBUG_MARKERS)
    return False


def _node_line_count(node: ast.AST) -> int:
    end = int(getattr(node, "end_lineno", getattr(node, "lineno", 0)) or 0)
    start = int(getattr(node, "lineno", end) or end)
    return max(1, end - start + 1)


def _function_quality_signal(text: str) -> int:
    return sum(
        marker in text
        for marker in ("def ", "class ", '"""', "'''", "return ", "raise ", "if ", "for ")
    )


def _messy_signal(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 8:
        return False
    long_lines = sum(len(line) > 100 for line in lines)
    bare_except = "except:" in text
    nested = any(line.startswith("            ") for line in lines)
    return bare_except or (long_lines / len(lines) > 0.25) or nested


def _trivial(text: str) -> bool:
    compact = [line.strip() for line in text.splitlines() if line.strip()]
    if len(compact) <= 3 and len(text) < 160:
        return True
    normalized = re.sub(r"\s+", "", text)
    return normalized in {"deff():pass", "defmain():pass"}
