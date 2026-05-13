from __future__ import annotations

from cw360.teacher.candidate import TeacherCandidate


def build_comments_prompt(candidate: TeacherCandidate) -> str:
    return (
        "Convert this source material into a Python comments or docstring example. The assistant "
        "response should improve clarity without changing behavior.\n\n"
        f"SOURCE_TEXT:\n{candidate.text}"
    )
