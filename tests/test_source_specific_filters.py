from __future__ import annotations

from cw360.data.dedupe import ExactDeduper
from cw360.data.schemas import QualityTier, TrainingExample
from cw360.data.source_filters import (
    chunk_python_around_defs,
    filter_codesearchnet_python,
    filter_debugbench,
    filter_educational_text,
    filter_stack_v2_python,
)


def test_stack_rejects_invalid_python_for_base_pretrain() -> None:
    example = TrainingExample(
        text="def broken(:\n    pass",
        source="stack_v2_python",
        example_type="python_code",
        training_stage="base_pretrain",
    )
    decision = filter_stack_v2_python(example)
    assert not decision.accepted
    assert decision.reason == "invalid_python_for_base_pretrain"


def test_stack_does_not_reject_unresolved_imports() -> None:
    example = TrainingExample(
        text="import made_up_package\n\ndef run(x):\n    return made_up_package.use(x)\n",
        source="stack_v2_python",
        example_type="python_code",
        training_stage="base_pretrain",
    )
    assert filter_stack_v2_python(example).accepted


def test_stack_marks_compact_utility_functions_for_candidate_mining() -> None:
    example = TrainingExample(
        text="def normalize_slug(value):\n    return value.strip().lower().replace(' ', '-')\n",
        source="stack_v2_python",
        example_type="python_code",
        training_stage="base_pretrain",
    )
    decision = filter_stack_v2_python(example)
    assert decision.quality_tier is QualityTier.TIER_A
    assert example.metadata["candidate_mining_signal"] == "compact_real_world_utility_function"


def test_stack_can_chunk_long_python_around_functions() -> None:
    text = "def a():\n    return 1\n\nclass B:\n    def c(self):\n        return 2\n"
    chunks = chunk_python_around_defs(text)
    assert any(chunk.startswith("def a") for chunk in chunks)
    assert any(chunk.startswith("class B") for chunk in chunks)


def test_codesearchnet_rejects_empty_or_useless_docstring() -> None:
    example = TrainingExample(
        text='"""TODO"""\ndef f(x):\n    return x\n',
        source="codesearchnet_python",
        example_type="docstring_code_pair",
        metadata={"docstring": "TODO"},
    )
    decision = filter_codesearchnet_python(example)
    assert not decision.accepted
    assert decision.reason == "empty_or_useless_docstring"


def test_debugbench_preserves_broken_code_with_traceback() -> None:
    example = TrainingExample(
        text="Traceback: SyntaxError\nbuggy_code:\ndef broken(:\n    pass\n",
        source="debugbench",
        example_type="debugging_case",
    )
    decision = filter_debugbench(example)
    assert decision.accepted
    assert example.metadata["candidate_pool"] == "debugging"


def test_debugbench_still_rejects_exact_duplicates() -> None:
    deduper = ExactDeduper()
    first = TrainingExample(
        text="Traceback: ValueError\nbuggy_code:\nint('x')\n",
        source="debugbench",
        example_type="debugging_case",
    )
    duplicate = TrainingExample(
        text="Traceback: ValueError\nbuggy_code:\nint('x')\n",
        source="debugbench",
        example_type="debugging_case",
    )
    assert filter_debugbench(first, deduper=deduper).accepted
    decision = filter_debugbench(duplicate, deduper=deduper)
    assert not decision.accepted
    assert decision.reason == "duplicate_content"


def test_educational_sources_do_not_ast_parse_and_keep_algorithm_text() -> None:
    example = TrainingExample(
        text=(
            "This lesson explains a graph algorithm and Python complexity "
            "without code syntax {{{."
        ),
        source="fineweb_edu",
        example_type="educational_text",
    )
    decision = filter_educational_text(example)
    assert decision.accepted
    assert decision.quality_tier is QualityTier.TIER_A
