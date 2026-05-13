from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from cw360.data.schemas import TrainingExample


class TokenizerLike(Protocol):
    eos_token: str | None
    eos_token_id: int | None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


@dataclass(frozen=True, slots=True)
class TokenizedExample:
    token_ids: list[int]
    source: str
    example_type: str
    metadata: dict[str, Any] = field(default_factory=dict)
    quality_tier: str | None = None
    loss_weight: float | None = None
    training_stage: str | None = None
    text: str | None = None

    @property
    def length(self) -> int:
        return len(self.token_ids)


def eos_token_id(tokenizer: TokenizerLike) -> int:
    token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(token_id, int):
        return token_id

    eos_token = getattr(tokenizer, "eos_token", None)
    if not isinstance(eos_token, str) or not eos_token:
        raise ValueError("tokenizer must provide eos_token_id or eos_token")

    encoded = tokenizer.encode(eos_token, add_special_tokens=False)
    if len(encoded) != 1:
        raise ValueError("tokenizer eos_token must encode to exactly one token")
    return int(encoded[0])


def append_eos_if_needed(token_ids: Sequence[int], eos_id: int) -> list[int]:
    output = [int(token_id) for token_id in token_ids]
    if not output or output[-1] != eos_id:
        output.append(eos_id)
    return output


def tokenize_example(
    example: TrainingExample,
    tokenizer: TokenizerLike,
    *,
    append_eos: bool = True,
    preserve_text: bool = False,
) -> TokenizedExample:
    """Tokenize a training example without normalizing whitespace or code text."""

    raw_token_ids = tokenizer.encode(example.text, add_special_tokens=False)
    token_ids = [int(token_id) for token_id in raw_token_ids]
    if append_eos:
        token_ids = append_eos_if_needed(token_ids, eos_token_id(tokenizer))

    return TokenizedExample(
        token_ids=token_ids,
        source=example.source,
        example_type=example.example_type,
        metadata=dict(example.metadata),
        quality_tier=example.quality_tier,
        loss_weight=example.loss_weight,
        training_stage=example.training_stage,
        text=example.text if preserve_text else None,
    )


def tokenize_examples(
    examples: Iterable[TrainingExample],
    tokenizer: TokenizerLike,
    *,
    append_eos: bool = True,
    preserve_text: bool = False,
) -> list[TokenizedExample]:
    return [
        tokenize_example(
            example,
            tokenizer,
            append_eos=append_eos,
            preserve_text=preserve_text,
        )
        for example in examples
    ]
