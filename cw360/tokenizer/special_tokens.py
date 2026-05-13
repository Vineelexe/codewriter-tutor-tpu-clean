from __future__ import annotations

from collections.abc import Iterable

DEFAULT_TOKENIZER_ID = "bigcode/starcoder2-15b"
UINT16_TOKEN_ID_LIMIT = 65_536
UINT16_MAX_TOKEN_ID = UINT16_TOKEN_ID_LIMIT - 1

FIM_TOKEN_MARKERS = (
    "fim",
    "<fim_prefix>",
    "<fim_middle>",
    "<fim_suffix>",
    "<fim_pad>",
)


def is_starcoder2_tokenizer_id(tokenizer_id: str) -> bool:
    return "starcoder2" in tokenizer_id.lower()


def collect_special_tokens(tokenizer: object) -> list[str]:
    tokens: list[str] = []

    all_special_tokens = getattr(tokenizer, "all_special_tokens", None)
    if isinstance(all_special_tokens, list):
        tokens.extend(str(token) for token in all_special_tokens)

    special_tokens_map = getattr(tokenizer, "special_tokens_map", None)
    if isinstance(special_tokens_map, dict):
        for value in special_tokens_map.values():
            if isinstance(value, str):
                tokens.append(value)
            elif isinstance(value, Iterable):
                tokens.extend(str(token) for token in value)

    return sorted(set(tokens))


def detect_fim_tokens(tokenizer: object) -> list[str]:
    candidates = set(collect_special_tokens(tokenizer))

    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        vocab = get_vocab()
        if isinstance(vocab, dict):
            candidates.update(str(token) for token in vocab)

    fim_tokens = [
        token
        for token in candidates
        if any(marker in token.lower() for marker in FIM_TOKEN_MARKERS)
    ]
    return sorted(fim_tokens)
