from __future__ import annotations

from cw360.teacher.candidate import TeacherCandidate


def build_tests_prompt(candidate: TeacherCandidate) -> str:
    return (
        "Convert this source material into a Python test-writing example. The assistant response "
        "should produce focused pytest-style coverage for normal and edge behavior.\n\n"
        f"SOURCE_TEXT:\n{candidate.text}"
    )
