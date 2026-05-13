from __future__ import annotations

from cw360.teacher.candidate import TeacherCandidate


def build_refactor_prompt(candidate: TeacherCandidate) -> str:
    return (
        "Convert this source material into a Python refactoring example. The assistant response "
        "should preserve behavior, improve readability, and avoid introducing dependencies.\n\n"
        f"SOURCE_TEXT:\n{candidate.text}"
    )
