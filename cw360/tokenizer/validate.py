from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from transformers import PreTrainedTokenizerBase

from cw360.tokenizer.loader import ensure_pad_token_for_batching, load_tokenizer
from cw360.tokenizer.special_tokens import (
    UINT16_TOKEN_ID_LIMIT,
    collect_special_tokens,
    detect_fim_tokens,
)

PYTHON_SAMPLE = """def add(a: int, b: int) -> int:
    total = a + b
    if total == 0:
        return total
    return total ** 2
"""

DOCSTRING_SAMPLE = '''def explain(value: str) -> str:
    """Return a normalized value.

    Args:
        value: Raw user input.

    Returns:
        A stripped value.
    """
    return value.strip()
'''


@dataclass(frozen=True)
class RoundTripCheck:
    name: str
    token_count: int
    indentation_survives: bool
    operators_survive: bool


@dataclass(frozen=True)
class TokenizerValidationReport:
    tokenizer_id: str
    vocab_size: int
    max_token_id: int
    eos_token: str | None
    bos_token: str | None
    pad_token: str | None
    pad_token_id: int | None
    pad_token_was_set_for_batching: bool
    special_tokens: list[str]
    fim_tokens: list[str]
    roundtrip_checks: list[RoundTripCheck]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def tokenizer_vocab(tokenizer: PreTrainedTokenizerBase) -> dict[str, int]:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        vocab = get_vocab()
        if isinstance(vocab, dict) and vocab:
            return {str(token): int(token_id) for token, token_id in vocab.items()}
    fallback_vocab_size = int(tokenizer.vocab_size)
    return {str(token_id): token_id for token_id in range(fallback_vocab_size)}


def vocab_size(tokenizer: PreTrainedTokenizerBase) -> int:
    configured_size = getattr(tokenizer, "vocab_size", None)
    if configured_size is not None:
        return int(configured_size)
    return len(tokenizer_vocab(tokenizer))


def max_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    vocab = tokenizer_vocab(tokenizer)
    return max(vocab.values())


def assert_uint16_safe(tokenizer: PreTrainedTokenizerBase) -> None:
    highest_id = max_token_id(tokenizer)
    if highest_id >= UINT16_TOKEN_ID_LIMIT:
        raise ValueError(
            "tokenizer max token id must be < "
            f"{UINT16_TOKEN_ID_LIMIT} for uint16 prepacked shards; got {highest_id}"
        )


def _encode_decode(tokenizer: PreTrainedTokenizerBase, text: str) -> tuple[list[int], str]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
    return list(token_ids), decoded


def _indentation_survives(original: str, decoded: str) -> bool:
    if "\n    " not in original:
        return True
    return "\n    " in decoded.replace("\r\n", "\n")


def _operators_survive(original: str, decoded: str) -> bool:
    operators = ("==", "+", "**", ":", "->", ".")
    required = [operator for operator in operators if operator in original]
    return all(operator in decoded for operator in required)


def roundtrip_check(tokenizer: PreTrainedTokenizerBase, name: str, text: str) -> RoundTripCheck:
    token_ids, decoded = _encode_decode(tokenizer, text)
    return RoundTripCheck(
        name=name,
        token_count=len(token_ids),
        indentation_survives=_indentation_survives(text, decoded),
        operators_survive=_operators_survive(text, decoded),
    )


def validate_loaded_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
    tokenizer_id: str,
    *,
    require_pad_for_batching: bool = False,
) -> TokenizerValidationReport:
    pad_token_was_set = False
    if require_pad_for_batching:
        pad_token_was_set = ensure_pad_token_for_batching(tokenizer)

    assert_uint16_safe(tokenizer)

    checks = [
        roundtrip_check(tokenizer, "python_code", PYTHON_SAMPLE),
        roundtrip_check(tokenizer, "docstring_heavy_python_code", DOCSTRING_SAMPLE),
    ]
    failed_checks = [
        check.name
        for check in checks
        if not check.indentation_survives or not check.operators_survive
    ]
    if failed_checks:
        raise ValueError(
            "tokenizer roundtrip lost required Python formatting/operators for: "
            + ", ".join(failed_checks)
        )

    return TokenizerValidationReport(
        tokenizer_id=tokenizer_id,
        vocab_size=vocab_size(tokenizer),
        max_token_id=max_token_id(tokenizer),
        eos_token=tokenizer.eos_token,
        bos_token=tokenizer.bos_token,
        pad_token=tokenizer.pad_token,
        pad_token_id=tokenizer.pad_token_id,
        pad_token_was_set_for_batching=pad_token_was_set,
        special_tokens=collect_special_tokens(tokenizer),
        fim_tokens=detect_fim_tokens(tokenizer),
        roundtrip_checks=checks,
    )


def validate_tokenizer(
    tokenizer_id: str,
    *,
    require_pad_for_batching: bool = False,
) -> TokenizerValidationReport:
    tokenizer = load_tokenizer(tokenizer_id)
    return validate_loaded_tokenizer(
        tokenizer,
        tokenizer_id,
        require_pad_for_batching=require_pad_for_batching,
    )
