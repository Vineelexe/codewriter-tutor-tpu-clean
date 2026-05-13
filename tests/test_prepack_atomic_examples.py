from __future__ import annotations

from cw360.data.tokenize import TokenizedExample
from cw360.prepack.writer import iter_prepacked_sequences


def test_atomic_instruction_example_is_not_split_across_sequences() -> None:
    tokenized = [
        TokenizedExample(
            [1, 2, 3, 4, 0],
            source="synthetic",
            example_type="instruction_debugging",
            metadata={"source_candidate_id": "debug-1", "task_type": "debugging"},
            quality_tier="A",
            training_stage="instruction_tune",
        )
    ]

    sequences = list(iter_prepacked_sequences(tokenized, max_seq_len=4))

    assert len(sequences) == 1
    assert sequences[0].token_ids == [1, 2, 3, 4, 0]
    assert sequences[0].packing_mode_counts == {"example_atomic_pack": 1}
    assert sequences[0].atomic_boundary_violations == 0


def test_continuous_pack_cuts_fixed_plus_one_windows() -> None:
    tokenized = [
        TokenizedExample(
            [1, 2, 3],
            source="stack_v2_python",
            example_type="python_code",
            metadata={"stable_input_hash": "a"},
        ),
        TokenizedExample(
            [4, 5, 6],
            source="stack_v2_python",
            example_type="python_code",
            metadata={"stable_input_hash": "b"},
        ),
    ]

    sequences = list(iter_prepacked_sequences(tokenized, max_seq_len=4))

    assert [sequence.token_ids for sequence in sequences] == [[1, 2, 3, 4, 5]]
    assert sequences[0].packing_mode_counts == {"continuous_pack": 1}
