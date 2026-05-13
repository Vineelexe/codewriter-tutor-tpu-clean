from __future__ import annotations

from cw360.teacher.candidate_router import CandidateRouter, TeacherCandidateConfig


def _config() -> TeacherCandidateConfig:
    return TeacherCandidateConfig(
        min_score=0.70,
        min_input_tokens=10,
        max_input_tokens=1600,
        max_candidates_per_source={"debugbench": 1, "stack_v2_python": 5},
        max_candidates_per_task_type={"debugging": 1, "comments": 1, "code_writing": 1},
        task_mix={
            "code_writing": 0.22,
            "debugging": 0.28,
            "explanation": 0.22,
            "comments": 0.10,
            "refactor": 0.08,
            "tests": 0.10,
        },
    )


def test_router_routes_debugbench_to_debugging_and_enforces_source_cap() -> None:
    router = CandidateRouter(_config(), max_candidates=10)
    row = {
        "error_message": "ValueError",
        "stack_trace": "Traceback (most recent call last): ValueError",
        "buggy_code": "def parse(value):\n    return int(value)\n",
        "fixed_code": (
            "def parse(value):\n"
            "    try:\n"
            "        return int(value)\n"
            "    except ValueError:\n"
            "        return 0\n"
        ),
    }

    first = router.route_example(row, source="debugbench")
    second = router.route_example(row, source="debugbench")

    assert first is not None
    assert first.task_type == "debugging"
    assert first.source == "debugbench"
    assert second is None
    assert router.source_counts["debugbench"] == 1


def test_router_falls_back_when_preferred_task_bucket_is_full() -> None:
    router = CandidateRouter(_config(), max_candidates=10)
    first = router.route_example(
        {"content": "def clamp(value, lower, upper):\n    return max(lower, min(value, upper))\n"},
        source="stack_v2_python",
    )
    second = router.route_example(
        {"content": "def normalize(value):\n    return value.strip().lower()\n"},
        source="stack_v2_python",
    )

    assert first is not None
    assert first.task_type == "comments"
    assert second is not None
    assert second.task_type != "comments"
    assert router.task_counts["comments"] == 1


def test_router_derives_task_caps_from_mix_when_explicit_caps_are_absent() -> None:
    config = TeacherCandidateConfig(
        min_score=0.70,
        min_input_tokens=10,
        max_candidates_per_source={"stack_v2_python": 10},
        task_mix={"comments": 0.10, "code_writing": 0.90},
    )
    router = CandidateRouter(config, max_candidates=5)

    routed = list(
        router.route_many(
            [
                {"content": "def one(value):\n    return value + 1\n"},
                {"content": "def two(value):\n    return value + 2\n"},
            ],
            source="stack_v2_python",
        )
    )

    assert len(routed) == 2
    assert router.task_counts["comments"] == 1
