from __future__ import annotations

from cw360.teacher.candidate import TeacherCandidate


def build_code_writing_prompt(candidate: TeacherCandidate) -> str:
    return (
        "Convert this source material into a Python code-writing User/Assistant example. "
        "The assistant response should include a correct, focused Python solution and concise "
        "teaching notes when useful.\n\n"
        f"SOURCE_TEXT:\n{candidate.text}"
    )
