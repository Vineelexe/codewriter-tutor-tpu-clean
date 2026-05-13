from __future__ import annotations

from cw360.data.extractors import (
    extract_codesearchnet_python,
    extract_debugbench,
    extract_fineweb_edu,
    extract_stack_v2_python,
)


def test_stack_v2_extractor_maps_content_to_training_example() -> None:
    example = extract_stack_v2_python(
        {"content": "def add(a, b):\n    return a + b\n", "path": "math.py"}
    )
    assert example.source == "stack_v2_python"
    assert example.example_type == "python_code"
    assert example.training_stage == "base_pretrain"
    assert example.metadata["path"] == "math.py"


def test_codesearchnet_extractor_preserves_docstring() -> None:
    example = extract_codesearchnet_python(
        {
            "code": "def area(radius):\n    return radius * radius\n",
            "docstring": "Compute circle area.",
        }
    )
    assert '"""Compute circle area."""' in example.text
    assert example.metadata["docstring"] == "Compute circle area."


def test_debugbench_extractor_preserves_error_trace_bug_and_fix() -> None:
    example = extract_debugbench(
        {
            "error_message": "ZeroDivisionError: division by zero",
            "traceback": "Traceback (most recent call last): ...",
            "buggy_code": "x = 1 / 0",
            "fixed_code": "if y != 0: x = 1 / y",
        }
    )
    assert "ZeroDivisionError" in example.text
    assert "buggy_code" in example.text
    assert example.source == "debugbench"


def test_fineweb_extractor_keeps_metadata() -> None:
    example = extract_fineweb_edu({"text": "Python functions can explain algorithms.", "id": "abc"})
    assert example.source == "fineweb_edu"
    assert example.metadata["id"] == "abc"
