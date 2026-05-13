from __future__ import annotations

from cw360.teacher.candidate import TeacherCandidate


def build_debugging_prompt(candidate: TeacherCandidate) -> str:
    return (
        "Convert this source material into a Python debugging example. The user prompt should "
        "show the failure clearly. The assistant response should explain the bug, provide a "
        "safe fix, and mention a focused test.\n\n"
        f"SOURCE_TEXT:\n{candidate.text}"
    )
