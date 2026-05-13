from __future__ import annotations

import pytest

from cw360.data.instruction import format_debugging_instruction, format_instruction


def test_instruction_format_is_stable_and_appends_eos_after_assistant() -> None:
    formatted = format_instruction(
        "Write a Python add function.",
        "def add(a, b):\n    return a + b\n",
        task_type="code_writing",
        eos_token="<eos>",
    )

    assert formatted.text == (
        "User:\n"
        "Write a Python add function.\n\n"
        "Assistant:\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "<eos>"
    )


def test_instruction_formatter_supports_debugging_task() -> None:
    formatted = format_debugging_instruction(
        "def div(a, b):\n    return a / b\n",
        "ZeroDivisionError",
        "def div(a, b):\n    return None if b == 0 else a / b\n",
    )

    assert formatted.task_type == "debugging"
    assert "ZeroDivisionError" in formatted.text
    assert "Assistant:\ndef div" in formatted.text


def test_instruction_formatter_rejects_general_chat_task() -> None:
    with pytest.raises(ValueError, match="unsupported instruction"):
        format_instruction("How is the weather?", "Sunny", task_type="general_chat")
