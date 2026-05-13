from __future__ import annotations

from cw360.teacher.candidate import TeacherCandidate


def build_explanation_prompt(candidate: TeacherCandidate) -> str:
    return (
        "Convert this source material into a Python explanation example. The assistant response "
        "should explain behavior, inputs, outputs, and key language features without broad "
        "chat.\n\n"
        f"SOURCE_TEXT:\n{candidate.text}"
    )
