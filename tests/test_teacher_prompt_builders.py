from __future__ import annotations

import json

from cw360.teacher.candidate import TeacherCandidate
from cw360.teacher.prompts import build_teacher_prompt


def _candidate(task_type: str = "debugging") -> TeacherCandidate:
    return TeacherCandidate(
        candidate_id="cand-1",
        source="debugbench",
        task_type=task_type,
        score=0.9,
        input_hash="abc",
        estimated_tokens=32,
        reason_selected="unit-test",
        quality_tier="A",
        proposed_train_stage="instruction_tune",
        proposed_loss_weight=1.0,
        raw_text="def divide(a, b):\n    return a / b\n",
    )


def test_prompt_builder_embeds_candidate_json_and_python_scope() -> None:
    bundle = build_teacher_prompt(_candidate("debugging"))

    assert "Python" in bundle.system_prompt
    assert "CANDIDATE_JSON:" in bundle.prompt
    assert '"candidate_id": "cand-1"' in bundle.prompt
    assert "Return JSON only" in bundle.prompt
    schema = json.loads(
        bundle.prompt.split("SCHEMA_HINT:", 1)[1]
        .split("\n\nTASK_INSTRUCTIONS:", 1)[0]
        .strip()
    )
    assert schema["required_content_fields"] == [
        "difficulty",
        "skills",
        "user_prompt",
        "assistant_response",
    ]
    schema_text = json.dumps(schema, sort_keys=True)
    for trusted_field in (
        "synthetic_id",
        "source_candidate_id",
        "teacher_provider",
        "teacher_model",
        "output_hash",
    ):
        assert trusted_field not in schema_text
