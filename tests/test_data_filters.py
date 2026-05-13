from __future__ import annotations

from cw360.data.dedupe import ExactDeduper
from cw360.data.filters import general_quality_filter
from cw360.data.schemas import TrainingExample


def test_empty_text_is_rejected() -> None:
    decision = general_quality_filter(
        TrainingExample(text="", source="synthetic", example_type="x")
    )
    assert not decision.accepted
    assert decision.reason == "empty_text"


def test_duplicate_content_is_rejected_by_exact_hash() -> None:
    deduper = ExactDeduper()
    example = TrainingExample(
        text="def add(a, b):\n    return a + b\n", source="synthetic", example_type="x"
    )
    assert general_quality_filter(example, deduper=deduper).accepted
    duplicate = TrainingExample(
        text="def add(a, b):\n    return a + b\n", source="synthetic", example_type="x"
    )
    decision = general_quality_filter(duplicate, deduper=deduper)
    assert decision.reason == "duplicate_content"


def test_dependency_dump_is_rejected() -> None:
    text = "\n".join(f"pkg{i}==1.0.{i}" for i in range(25))
    example = TrainingExample(text=text, source="stack_v2_python", example_type="python_code")
    decision = general_quality_filter(example)
    assert decision.reason == "lockfile_or_dependency_dump"


def test_long_programming_example_is_rescued_not_blindly_dropped() -> None:
    text = "\n".join(f"def f{i}():\n    return {i}" for i in range(5000))
    example = TrainingExample(text=text, source="synthetic", example_type="python_code")
    decision = general_quality_filter(example)
    assert decision.rescue
    assert decision.status == "rescued"
