from __future__ import annotations

import ast
import re
from collections.abc import Iterable

from cw360.data.dedupe import ExactDeduper
from cw360.data.filters import (
    estimate_token_count,
    general_quality_filter,
    has_programming_signal,
    has_repetition_spam,
    has_very_long_unbroken_line,
    obvious_trash_filter,
)
from cw360.data.quality_tiers import tier_from_signal
from cw360.data.schemas import FilterDecision, TrainingExample

_USELESS_DOCSTRINGS = {"", "todo", "tbd", "none", "n/a", "fixme", "function", "method"}
_CS_TERMS = (
    "python",
    "code",
    "programming",
    "algorithm",
    "data structure",
    "debug",
    "function",
    "class",
    "complexity",
    "recursion",
    "compiler",
)


def parse_python(text: str) -> ast.AST | None:
    try:
        return ast.parse(text)
    except SyntaxError:
        return None


def strip_massive_header_comments(text: str, *, max_header_lines: int = 40) -> tuple[str, bool]:
    lines = text.splitlines()
    header_end = 0
    for line in lines:
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith("#")
            or stripped.startswith('"""')
            or stripped.startswith("'''")
        ):
            header_end += 1
            continue
        break
    if header_end > max_header_lines:
        return "\n".join(lines[header_end:]).lstrip(), True
    return text, False


def compact_utility_function_signal(tree: ast.AST) -> bool:
    functions = [
        node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not functions:
        return False
    for func in functions:
        if len(func.body) <= 30 and len(func.args.args) <= 6:
            return True
    return False


def chunk_python_around_defs(text: str, *, max_chars: int = 12000) -> list[str]:
    tree = parse_python(text)
    if tree is None:
        return [text]
    lines = text.splitlines()
    chunks: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and hasattr(
            node, "end_lineno"
        ):
            start = max(0, node.lineno - 1)
            end = int(node.end_lineno or node.lineno)
            chunk = "\n".join(lines[start:end])
            if chunk and len(chunk) <= max_chars:
                chunks.append(chunk)
    return chunks or [text]


def filter_stack_v2_python(
    example: TrainingExample, *, deduper: ExactDeduper | None = None
) -> FilterDecision:
    text, stripped_header = strip_massive_header_comments(example.text)
    if stripped_header:
        example.text = text
        example.metadata["massive_header_stripped"] = True
    trash = obvious_trash_filter(example, deduper=deduper)
    if trash is not None:
        return trash
    tree = parse_python(example.text)
    if tree is None:
        if example.training_stage == "base_pretrain" and not example.metadata.get(
            "debugging_reason"
        ):
            return tier_from_signal(trash=True, reason="invalid_python_for_base_pretrain")
        return tier_from_signal(rescue=True, reason="invalid_python_but_debugging_relevant")
    if estimate_token_count(example.text) > 8192:
        return tier_from_signal(
            rescue=True, reason="long_python_file_chunk_around_functions_or_classes"
        )
    if compact_utility_function_signal(tree):
        example.metadata["candidate_mining_signal"] = "compact_real_world_utility_function"
        return tier_from_signal(high_signal=True, reason="valid_python_compact_utility_function")
    return tier_from_signal(useful=True, reason="valid_python")


def _docstring_from_example(example: TrainingExample) -> str:
    docstring = example.metadata.get("docstring")
    if isinstance(docstring, str):
        return docstring.strip()
    tree = parse_python(example.text)
    if tree is None:
        return ""
    return ast.get_docstring(tree) or ""


def filter_codesearchnet_python(
    example: TrainingExample, *, deduper: ExactDeduper | None = None
) -> FilterDecision:
    trash = obvious_trash_filter(example, deduper=deduper)
    if trash is not None:
        return trash
    if parse_python(example.text) is None:
        return tier_from_signal(trash=True, reason="invalid_python_docstring_code_pair")
    docstring = _docstring_from_example(example)
    normalized_docstring = re.sub(r"\W+", " ", docstring).strip().lower()
    if normalized_docstring in _USELESS_DOCSTRINGS or len(normalized_docstring.split()) < 3:
        return tier_from_signal(trash=True, reason="empty_or_useless_docstring")
    code_len = max(1, len(example.text) - len(docstring))
    doc_density = len(docstring) / code_len
    if doc_density > 2.0:
        return tier_from_signal(useful=True, reason="docstring_heavy_but_valid")
    return tier_from_signal(high_signal=True, reason="dense_docstring_code_pair")


def filter_debugbench(
    example: TrainingExample, *, deduper: ExactDeduper | None = None
) -> FilterDecision:
    trash = obvious_trash_filter(example, deduper=deduper)
    if trash is not None:
        return trash
    lowered = example.text.lower()
    has_debug_signal = any(
        marker in lowered
        for marker in ("traceback", "stack_trace", "error", "exception", "buggy_code", "fixed_code")
    )
    if has_debug_signal:
        example.metadata["candidate_pool"] = "debugging"
        return tier_from_signal(high_signal=True, reason="debugging_case_preserve_broken_code")
    return tier_from_signal(useful=True, reason="debugbench_case_without_explicit_trace")


def filter_educational_text(
    example: TrainingExample, *, deduper: ExactDeduper | None = None
) -> FilterDecision:
    trash = obvious_trash_filter(example, deduper=deduper)
    if trash is not None:
        return trash
    if has_very_long_unbroken_line(example.text):
        return tier_from_signal(trash=True, reason="very_long_unbroken_line")
    if has_repetition_spam(example.text):
        return tier_from_signal(trash=True, reason="repetition_spam")
    lowered = example.text.lower()
    cs_hits = sum(term in lowered for term in _CS_TERMS)
    if cs_hits >= 2 and has_programming_signal(example.text):
        return tier_from_signal(high_signal=True, reason="cs_programming_educational_text")
    if cs_hits >= 1:
        return tier_from_signal(useful=True, reason="possibly_relevant_educational_text")
    return tier_from_signal(trash=True, reason="non_cs_or_non_educational_text")


def filter_synthetic(
    example: TrainingExample, *, deduper: ExactDeduper | None = None
) -> FilterDecision:
    trash = obvious_trash_filter(example, deduper=deduper)
    if trash is not None:
        return trash
    if not isinstance(example.metadata, dict):
        return tier_from_signal(trash=True, reason="malformed_synthetic_metadata")
    if has_programming_signal(example.text):
        return tier_from_signal(
            useful=True, reason="synthetic_programming_example_pending_phase11_validation"
        )
    return tier_from_signal(useful=True, reason="synthetic_example_pending_phase11_validation")


SOURCE_FILTERS = {
    "stack_v2_python": filter_stack_v2_python,
    "codesearchnet_python": filter_codesearchnet_python,
    "debugbench": filter_debugbench,
    "fineweb_edu": filter_educational_text,
    "cosmopedia_cs": filter_educational_text,
    "synthetic": filter_synthetic,
}


def filter_training_example(
    example: TrainingExample, *, deduper: ExactDeduper | None = None
) -> FilterDecision:
    source_filter = SOURCE_FILTERS.get(example.source)
    if source_filter is None:
        return general_quality_filter(example, deduper=deduper)
    return source_filter(example, deduper=deduper)


def filter_many(
    examples: Iterable[TrainingExample],
    *,
    deduper: ExactDeduper | None = None,
) -> list[tuple[TrainingExample, FilterDecision]]:
    active_deduper = deduper or ExactDeduper()
    return [
        (example, filter_training_example(example, deduper=active_deduper)) for example in examples
    ]
