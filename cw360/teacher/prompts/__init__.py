from __future__ import annotations

from dataclasses import dataclass

from cw360.teacher.candidate import TeacherCandidate
from cw360.teacher.prompts.code_writing import build_code_writing_prompt
from cw360.teacher.prompts.comments import build_comments_prompt
from cw360.teacher.prompts.data_structuring import build_data_structuring_prompt
from cw360.teacher.prompts.debugging import build_debugging_prompt
from cw360.teacher.prompts.explanation import build_explanation_prompt
from cw360.teacher.prompts.refactor import build_refactor_prompt
from cw360.teacher.prompts.tests import build_tests_prompt

SYSTEM_PROMPT = (
    "You are a Python coding tutor data structuring engine. Return one strict JSON object "
    "for a Python-only CodeWriter-Tutor SLM training record. Do not include markdown."
)


@dataclass(frozen=True, slots=True)
class PromptBundle:
    system_prompt: str
    prompt: str


_BUILDERS = {
    "code_writing": build_code_writing_prompt,
    "debugging": build_debugging_prompt,
    "explanation": build_explanation_prompt,
    "comments": build_comments_prompt,
    "refactor": build_refactor_prompt,
    "tests": build_tests_prompt,
}


def build_teacher_prompt(
    candidate: TeacherCandidate,
    *,
    record_type: str = "structured",
) -> PromptBundle:
    task_prompt = _BUILDERS.get(candidate.task_type, build_code_writing_prompt)(candidate)
    return PromptBundle(
        system_prompt=SYSTEM_PROMPT,
        prompt=build_data_structuring_prompt(candidate, task_prompt, record_type=record_type),
    )


__all__ = ["PromptBundle", "SYSTEM_PROMPT", "build_teacher_prompt"]
