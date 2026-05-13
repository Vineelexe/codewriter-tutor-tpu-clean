from __future__ import annotations

import json

from cw360.teacher.candidate import TeacherCandidate


def build_data_structuring_prompt(
    candidate: TeacherCandidate,
    task_prompt: str,
    *,
    record_type: str = "structured",
) -> str:
    schema = {
        "required_content_fields": ["difficulty", "skills", "user_prompt", "assistant_response"],
        "optional_content_fields": [
            "code_before",
            "code_after",
            "explanation",
            "tests",
            "quality_notes",
        ],
        "difficulty": ["beginner", "intermediate", "advanced"],
        "example": {
            "difficulty": "beginner",
            "skills": ["python", "debugging", "edge_cases"],
            "user_prompt": "Debug this Python function...",
            "assistant_response": "The Python bug is...",
            "code_before": "def example(): ...",
            "code_after": "def example(): ...",
            "explanation": "The fix works because...",
            "tests": "def test_example(): ...",
            "quality_notes": "Focused on a concrete Python edge case.",
        },
    }
    return (
        "Create one canonical Python training record. Keep it specialized for Python code "
        "writing, debugging, and tutoring. Return JSON only with content fields. The "
        "engine will stamp provenance, provider, model, hashes, curriculum, and metadata.\n\n"
        f"SCHEMA_HINT: {json.dumps(schema, sort_keys=True)}\n\n"
        f"TASK_INSTRUCTIONS:\n{task_prompt}\n\n"
        "CANDIDATE_JSON:\n"
        f"{json.dumps(candidate.to_dict(), sort_keys=True)}"
    )
