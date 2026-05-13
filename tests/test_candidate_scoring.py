from __future__ import annotations

from cw360.teacher.candidate_score import score_teacher_candidate


def test_scores_debugbench_exception_as_high_value_debugging_candidate() -> None:
    text = """error:
ZeroDivisionError

stack_trace:
Traceback (most recent call last):
  File "calc.py", line 3, in average
ZeroDivisionError: division by zero

buggy_code:
def average(values):
    return sum(values) / len(values)
"""

    score = score_teacher_candidate(source="debugbench", text=text, min_input_tokens=10)

    assert score.accepted
    assert score.quality_tier == "A"
    assert score.task_preferences[0] == "debugging"
    assert "debug_signal" in score.reason


def test_rejects_dependency_dumps_and_binary_like_text() -> None:
    dependency_dump = "\n".join(f"package{i}==1.0.{i}" for i in range(25))
    corrupt = "def ok():\n    return 1\n\x00\x00"

    assert not score_teacher_candidate(source="stack_v2_python", text=dependency_dump).accepted
    assert "dependency_dump" in score_teacher_candidate(
        source="stack_v2_python", text=dependency_dump
    ).reason
    assert not score_teacher_candidate(source="stack_v2_python", text=corrupt).accepted


def test_long_useful_python_is_excerpted_instead_of_size_rejected() -> None:
    filler = "\n".join(f"# explanatory filler {index}" for index in range(2500))
    useful = '''
def parse_int(value, default=0):
    """Parse an integer value with a safe fallback."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
'''
    text = f"{filler}\n{useful}\n{filler}"

    score = score_teacher_candidate(
        source="stack_v2_python",
        text=text,
        max_input_tokens=120,
        min_input_tokens=10,
    )

    assert score.accepted
    assert score.estimated_tokens <= 120
    assert "parse_int" in score.selected_text
    assert score.metadata["excerpted_from_long_input"] is True


def test_codesearchnet_docstring_prefers_explanation() -> None:
    text = '''"""Normalize a label for display in a URL."""
def slugify(label):
    cleaned = label.strip().lower()
    return "-".join(part for part in cleaned.split() if part)
'''

    score = score_teacher_candidate(
        source="codesearchnet_python",
        text=text,
        min_input_tokens=10,
    )

    assert score.accepted
    assert score.task_preferences[0] == "explanation"
