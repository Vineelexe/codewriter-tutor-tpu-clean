from __future__ import annotations

from cw360.data.packing import example_atomic_pack
from cw360.data.tokenize import TokenizedExample


def test_atomic_pack_does_not_split_instruction_examples_across_windows() -> None:
    first = TokenizedExample([1, 2, 0], source="synthetic", example_type="instruction")
    second = TokenizedExample([3, 4, 0], source="synthetic", example_type="debugging")
    third = TokenizedExample([5, 6, 7, 8, 0], source="synthetic", example_type="tests")

    sequences = example_atomic_pack([first, second, third], max_seq_len=4)

    assert [sequence.token_ids for sequence in sequences] == [[1, 2, 0], [3, 4, 0], [5, 6, 7, 8, 0]]
    assert [sequence.example_count for sequence in sequences] == [1, 1, 1]


def test_atomic_tpu_prepack_emits_full_sequences_only() -> None:
    sequences = example_atomic_pack(
        [[1, 2, 0], [3, 4, 5, 6, 0]],
        max_seq_len=4,
        tpu_prepack=True,
    )

    assert [sequence.token_ids for sequence in sequences] == [[3, 4, 5, 6, 0]]
    assert all(sequence.length == 5 for sequence in sequences)


def test_atomic_tpu_prepack_emits_combined_examples_that_exactly_fill_window() -> None:
    sequences = example_atomic_pack(
        [[1, 2], [3, 4, 0]],
        max_seq_len=4,
        tpu_prepack=True,
    )

    assert [sequence.token_ids for sequence in sequences] == [[1, 2, 3, 4, 0]]


def test_atomic_pack_truncates_long_fim_examples_while_preserving_markers() -> None:
    fim_example = TokenizedExample(
        [101, 1, 2, 3, 102, 4, 5, 6, 103, 7, 8, 9],
        source="synthetic",
        example_type="fim",
        metadata={"fim_marker_token_ids": {"prefix": 101, "suffix": 102, "middle": 103}},
    )

    sequences = example_atomic_pack([fim_example], max_seq_len=6)

    assert len(sequences) == 1
    assert len(sequences[0].token_ids) == 7
    assert 101 in sequences[0].token_ids
    assert 102 in sequences[0].token_ids
    assert 103 in sequences[0].token_ids
    assert sequences[0].token_ids.index(101) < sequences[0].token_ids.index(102)
    assert sequences[0].token_ids.index(102) < sequences[0].token_ids.index(103)
