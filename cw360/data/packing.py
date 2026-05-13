from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from cw360.data.tokenize import TokenizedExample

TPU_PREPACK_LOSS_NOTE = (
    "Real TPU prepack emits token-ID-only sequences with shape [N, max_seq_len + 1]. "
    "loss_weight is applied only during example sampling or oversampling before shard "
    "creation; the TPU trainer uses uniform causal LM loss unless a separate weighted "
    "shard format is explicitly implemented and tested."
)


@dataclass(frozen=True, slots=True)
class PackedTokenSequence:
    token_ids: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)
    example_count: int = 0

    @property
    def length(self) -> int:
        return len(self.token_ids)


def _coerce_tokens(example: TokenizedExample | Sequence[int]) -> list[int]:
    if isinstance(example, TokenizedExample):
        return list(example.token_ids)
    return [int(token_id) for token_id in example]


def _metadata_for(example: TokenizedExample | Sequence[int]) -> dict[str, Any]:
    if not isinstance(example, TokenizedExample):
        return {}
    return {
        "source": example.source,
        "example_type": example.example_type,
        "metadata": dict(example.metadata),
        "quality_tier": example.quality_tier,
        "loss_weight": example.loss_weight,
        "training_stage": example.training_stage,
    }


def _truncate_tokens(
    tokens: Sequence[int],
    window_size: int,
    metadata: dict[str, Any],
) -> list[int]:
    if len(tokens) <= window_size:
        return list(tokens)

    nested_metadata = metadata.get("metadata")
    marker_ids = metadata.get("fim_marker_token_ids")
    if not isinstance(marker_ids, dict) and isinstance(nested_metadata, dict):
        marker_ids = nested_metadata.get("fim_marker_token_ids")
    if not isinstance(marker_ids, dict):
        return list(tokens[:window_size])

    prefix_id = marker_ids.get("prefix")
    suffix_id = marker_ids.get("suffix")
    middle_id = marker_ids.get("middle")
    if not all(isinstance(token_id, int) for token_id in (prefix_id, suffix_id, middle_id)):
        return list(tokens[:window_size])

    try:
        prefix_pos = tokens.index(prefix_id)
        suffix_pos = tokens.index(suffix_id)
        middle_pos = tokens.index(middle_id)
    except ValueError:
        return list(tokens[:window_size])

    if not (prefix_pos < suffix_pos < middle_pos):
        return list(tokens[:window_size])

    prefix_part = list(tokens[prefix_pos + 1 : suffix_pos])
    suffix_part = list(tokens[suffix_pos + 1 : middle_pos])
    middle_part = list(tokens[middle_pos + 1 :])
    budget = window_size - 3
    if budget <= 0:
        return [prefix_id, suffix_id, middle_id][:window_size]

    allocations = _allocate_fim_budget(prefix_part, suffix_part, middle_part, budget)
    prefix_keep, suffix_keep, middle_keep = allocations
    return (
        [prefix_id]
        + prefix_part[:prefix_keep]
        + [suffix_id]
        + suffix_part[:suffix_keep]
        + [middle_id]
        + middle_part[:middle_keep]
    )


def _allocate_fim_budget(
    prefix_part: Sequence[int],
    suffix_part: Sequence[int],
    middle_part: Sequence[int],
    budget: int,
) -> tuple[int, int, int]:
    lengths = [len(prefix_part), len(suffix_part), len(middle_part)]
    kept = [0, 0, 0]
    remaining = budget

    while remaining > 0 and kept != lengths:
        progressed = False
        for index in range(3):
            if kept[index] < lengths[index] and remaining > 0:
                kept[index] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    return kept[0], kept[1], kept[2]


def continuous_pack(
    examples: Iterable[TokenizedExample | Sequence[int]],
    *,
    max_seq_len: int,
    tpu_prepack: bool = False,
    drop_remainder: bool | None = None,
) -> list[PackedTokenSequence]:
    """Concatenate examples and slice into `max_seq_len + 1` causal LM windows."""

    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")

    window_size = max_seq_len + 1
    should_drop_remainder = tpu_prepack if drop_remainder is None else drop_remainder
    stream: list[int] = []
    for example in examples:
        stream.extend(_coerce_tokens(example))

    sequences: list[PackedTokenSequence] = []
    for start in range(0, len(stream), window_size):
        window = stream[start : start + window_size]
        if len(window) < window_size and should_drop_remainder:
            break
        sequences.append(
            PackedTokenSequence(
                token_ids=window,
                metadata={"packing_mode": "continuous", "tpu_prepack": tpu_prepack},
            )
        )
    return sequences


def example_atomic_pack(
    examples: Iterable[TokenizedExample | Sequence[int]],
    *,
    max_seq_len: int,
    tpu_prepack: bool = False,
    pad_token_id: int | None = None,
) -> list[PackedTokenSequence]:
    """Pack whole examples into windows without splitting a logical example."""

    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if tpu_prepack and pad_token_id is not None:
        raise ValueError("real TPU prepack must not pad atomic sequences")

    window_size = max_seq_len + 1
    sequences: list[PackedTokenSequence] = []
    current: list[int] = []
    current_meta: list[dict[str, Any]] = []

    for example in examples:
        metadata = _metadata_for(example)
        original_tokens = _coerce_tokens(example)
        tokens = _truncate_tokens(original_tokens, window_size, metadata)
        if len(tokens) > window_size:
            raise AssertionError("atomic truncation failed to fit the target window")

        if current and len(current) + len(tokens) > window_size:
            if not tpu_prepack or len(current) == window_size:
                sequences.append(
                    PackedTokenSequence(
                        token_ids=current,
                        metadata={
                            "packing_mode": "example_atomic",
                            "examples": current_meta,
                            "tpu_prepack": tpu_prepack,
                        },
                        example_count=len(current_meta),
                    )
                )
            current = []
            current_meta = []

        if len(tokens) == window_size:
            sequences.append(
                PackedTokenSequence(
                    token_ids=tokens,
                    metadata={
                        "packing_mode": "example_atomic",
                        "examples": [metadata],
                        "tpu_prepack": tpu_prepack,
                        "truncated": len(original_tokens) > window_size,
                    },
                    example_count=1,
                )
            )
            continue

        current.extend(tokens)
        current_meta.append(metadata)
        if len(current) == window_size:
            sequences.append(
                PackedTokenSequence(
                    token_ids=current,
                    metadata={
                        "packing_mode": "example_atomic",
                        "examples": current_meta,
                        "tpu_prepack": tpu_prepack,
                    },
                    example_count=len(current_meta),
                )
            )
            current = []
            current_meta = []

    if current and not tpu_prepack:
        sequences.append(
            PackedTokenSequence(
                token_ids=current,
                metadata={
                    "packing_mode": "example_atomic",
                    "examples": current_meta,
                    "tpu_prepack": tpu_prepack,
                },
                example_count=len(current_meta),
            )
        )

    return sequences


def packed_to_lm_arrays(sequence: PackedTokenSequence) -> tuple[list[int], list[int]]:
    if len(sequence.token_ids) < 2:
        raise ValueError("packed causal LM sequence must contain at least two tokens")
    return list(sequence.token_ids[:-1]), list(sequence.token_ids[1:])
