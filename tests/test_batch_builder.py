from __future__ import annotations

from cw360.data.batching import build_causal_lm_batch
from cw360.data.packing import PackedTokenSequence


def test_batch_builder_pads_cpu_batch_and_masks_padding_labels() -> None:
    batch = build_causal_lm_batch(
        [
            PackedTokenSequence([10, 11, 12, 13]),
            PackedTokenSequence([20, 21]),
        ],
        seq_len=3,
        pad_token_id=0,
    )

    assert batch.input_ids.shape == (2, 3)
    assert batch.labels.shape == (2, 3)
    assert batch.attention_mask.tolist() == [[1, 1, 1], [1, 1, 0]]
    assert batch.input_ids.tolist() == [[10, 11, 12], [20, 21, 0]]
    assert batch.labels.tolist() == [[11, 12, 13], [21, -100, -100]]
