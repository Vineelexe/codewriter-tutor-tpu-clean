from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

from cw360.tokenizer.special_tokens import detect_fim_tokens

FALLBACK_FIM_PREFIX = "<CW360_FIM_PREFIX>"
FALLBACK_FIM_SUFFIX = "<CW360_FIM_SUFFIX>"
FALLBACK_FIM_MIDDLE = "<CW360_FIM_MIDDLE>"


class FIMTokenizerLike(Protocol):
    def get_vocab(self) -> dict[str, int]: ...


@dataclass(frozen=True, slots=True)
class FIMParts:
    prefix: str
    middle: str
    suffix: str


@dataclass(frozen=True, slots=True)
class FIMMarkers:
    prefix: str
    suffix: str
    middle: str
    tokenizer_supported: bool


@dataclass(frozen=True, slots=True)
class FIMFormattedText:
    text: str
    parts: FIMParts
    markers: FIMMarkers
    applied: bool


def split_fim_code(
    code: str,
    *,
    middle_start: int | None = None,
    middle_end: int | None = None,
) -> FIMParts:
    if middle_start is None or middle_end is None:
        first = len(code) // 3
        second = (len(code) * 2) // 3
        middle_start = _line_boundary_at_or_before(code, first)
        middle_end = _line_boundary_at_or_after(code, second)

    if middle_start < 0 or middle_end < middle_start or middle_end > len(code):
        raise ValueError("invalid FIM middle span")

    return FIMParts(
        prefix=code[:middle_start],
        middle=code[middle_start:middle_end],
        suffix=code[middle_end:],
    )


def _line_boundary_at_or_before(text: str, index: int) -> int:
    newline = text.rfind("\n", 0, index)
    return 0 if newline < 0 else newline + 1


def _line_boundary_at_or_after(text: str, index: int) -> int:
    newline = text.find("\n", index)
    return len(text) if newline < 0 else newline + 1


def resolve_fim_markers(tokenizer: object | None = None) -> FIMMarkers:
    tokens = detect_fim_tokens(tokenizer) if tokenizer is not None else []

    prefix = _find_marker(tokens, "prefix")
    suffix = _find_marker(tokens, "suffix")
    middle = _find_marker(tokens, "middle")
    if prefix and suffix and middle:
        return FIMMarkers(prefix=prefix, suffix=suffix, middle=middle, tokenizer_supported=True)

    return FIMMarkers(
        prefix=FALLBACK_FIM_PREFIX,
        suffix=FALLBACK_FIM_SUFFIX,
        middle=FALLBACK_FIM_MIDDLE,
        tokenizer_supported=False,
    )


def _find_marker(tokens: list[str], name: str) -> str | None:
    needle = name.lower()
    for token in tokens:
        if "fim" in token.lower() and needle in token.lower():
            return token
    return None


def format_fim_code(
    code: str,
    *,
    tokenizer: object | None = None,
    fim_probability: float = 1.0,
    rng: random.Random | None = None,
    middle_start: int | None = None,
    middle_end: int | None = None,
) -> FIMFormattedText:
    if not 0.0 <= fim_probability <= 1.0:
        raise ValueError("fim_probability must be between 0.0 and 1.0")

    random_source = rng or random.Random(0)
    if random_source.random() >= fim_probability:
        empty_parts = FIMParts(prefix=code, middle="", suffix="")
        return FIMFormattedText(
            text=code,
            parts=empty_parts,
            markers=resolve_fim_markers(tokenizer),
            applied=False,
        )

    parts = split_fim_code(code, middle_start=middle_start, middle_end=middle_end)
    markers = resolve_fim_markers(tokenizer)
    text = (
        f"{markers.prefix}{parts.prefix}"
        f"{markers.suffix}{parts.suffix}"
        f"{markers.middle}{parts.middle}"
    )
    return FIMFormattedText(text=text, parts=parts, markers=markers, applied=True)


def format_fim(*args: object, **kwargs: object) -> FIMFormattedText:
    return format_fim_code(*args, **kwargs)
