from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_INSTRUCTION_TASKS = frozenset(
    {
        "code_writing",
        "debugging",
        "explanation",
        "comments",
        "refactoring",
        "tests",
    }
)


@dataclass(frozen=True, slots=True)
class InstructionFormattedText:
    text: str
    task_type: str
    prompt: str
    response: str


def format_instruction(
    prompt: str,
    response: str,
    *,
    task_type: str = "code_writing",
    eos_token: str | None = None,
) -> InstructionFormattedText:
    if task_type not in SUPPORTED_INSTRUCTION_TASKS:
        raise ValueError(f"unsupported instruction task_type: {task_type}")

    suffix = eos_token or ""
    text = f"User:\n{prompt}\n\nAssistant:\n{response}{suffix}"
    return InstructionFormattedText(
        text=text,
        task_type=task_type,
        prompt=prompt,
        response=response,
    )


def format_debugging_instruction(
    buggy_code: str,
    error: str,
    fixed_code: str,
    *,
    eos_token: str | None = None,
) -> InstructionFormattedText:
    prompt = f"Debug this Python code.\n\nError:\n{error}\n\nCode:\n{buggy_code}"
    return format_instruction(
        prompt,
        fixed_code,
        task_type="debugging",
        eos_token=eos_token,
    )
