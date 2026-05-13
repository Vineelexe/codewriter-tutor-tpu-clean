from __future__ import annotations

from cw360.data.packing import TPU_PREPACK_LOSS_NOTE, continuous_pack


def test_tpu_prepack_emits_only_full_seq_plus_one_windows_without_padding() -> None:
    sequences = continuous_pack(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]],
        max_seq_len=4,
        tpu_prepack=True,
    )

    assert [sequence.token_ids for sequence in sequences] == [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    assert all(sequence.length == 5 for sequence in sequences)


def test_tpu_prepack_drops_incomplete_remainder_instead_of_padding() -> None:
    sequences = continuous_pack(
        [[1, 2, 3], [4, 5, 6]],
        max_seq_len=4,
        tpu_prepack=True,
    )

    assert [sequence.token_ids for sequence in sequences] == [[1, 2, 3, 4, 5]]
    assert "uniform causal LM loss" in TPU_PREPACK_LOSS_NOTE
    assert "loss_weight is applied only during example sampling" in TPU_PREPACK_LOSS_NOTE
