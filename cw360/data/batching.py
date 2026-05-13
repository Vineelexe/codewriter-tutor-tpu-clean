from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch

from cw360.data.packing import PackedTokenSequence


@dataclass(frozen=True, slots=True)
class CausalLMBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor


def build_causal_lm_batch(
    sequences: Iterable[PackedTokenSequence | Sequence[int]],
    *,
    seq_len: int,
    pad_token_id: int,
    ignore_index: int = -100,
    include_attention_mask: bool = True,
) -> CausalLMBatch:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")

    windows = [_coerce_window(sequence) for sequence in sequences]
    if not windows:
        raise ValueError("at least one packed sequence is required")

    padded_windows: list[list[int]] = []
    for window in windows:
        if len(window) > seq_len + 1:
            raise ValueError("packed sequence is longer than seq_len + 1")
        padded = list(window) + [pad_token_id] * (seq_len + 1 - len(window))
        padded_windows.append(padded)

    input_rows = [window[:-1] for window in padded_windows]
    label_rows = [window[1:] for window in padded_windows]
    mask_rows = [
        [1 if token_id != pad_token_id else 0 for token_id in input_row]
        for input_row in input_rows
    ]

    for row_index, original in enumerate(windows):
        valid_label_count = max(0, len(original) - 1)
        for label_index in range(valid_label_count, seq_len):
            label_rows[row_index][label_index] = ignore_index

    input_ids = torch.tensor(input_rows, dtype=torch.long)
    labels = torch.tensor(label_rows, dtype=torch.long)
    attention_mask = torch.tensor(mask_rows, dtype=torch.long)
    if not include_attention_mask:
        attention_mask = torch.ones_like(input_ids)

    return CausalLMBatch(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
    )


def _coerce_window(sequence: PackedTokenSequence | Sequence[int]) -> list[int]:
    if isinstance(sequence, PackedTokenSequence):
        return list(sequence.token_ids)
    return [int(token_id) for token_id in sequence]
