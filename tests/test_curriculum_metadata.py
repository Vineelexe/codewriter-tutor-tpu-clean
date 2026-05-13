from __future__ import annotations

from cw360.teacher.curriculum import curriculum_metadata, infer_skills
from tests.test_teacher_prompt_builders import _candidate


def test_curriculum_metadata_preserves_candidate_training_hints() -> None:
    metadata = curriculum_metadata(_candidate("debugging"))

    assert metadata["train_stage"] == "instruction_tune"
    assert metadata["loss_weight"] == 1.0
    assert "errors" in metadata["skills"]


def test_infer_skills_detects_python_testing_and_json() -> None:
    skills = infer_skills(
        "import json\n\ndef test_loads():\n    assert json.loads('{}') == {}",
        "tests",
    )

    assert "JSON" in skills
    assert "tests" in skills
