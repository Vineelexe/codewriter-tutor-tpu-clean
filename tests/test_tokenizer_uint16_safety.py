from __future__ import annotations

import pytest

from cw360.tokenizer.validate import assert_uint16_safe, max_token_id


class TinySafeTokenizer:
    vocab_size = 3

    def get_vocab(self) -> dict[str, int]:
        return {"a": 0, "b": 1, "c": 65_535}


class TooLargeTokenizer:
    vocab_size = 2

    def get_vocab(self) -> dict[str, int]:
        return {"a": 0, "overflow": 65_536}


def test_uint16_safety_allows_highest_uint16_token_id() -> None:
    tokenizer = TinySafeTokenizer()

    assert max_token_id(tokenizer) == 65_535
    assert_uint16_safe(tokenizer)


def test_uint16_safety_rejects_token_ids_over_limit() -> None:
    with pytest.raises(ValueError, match="must be < 65536"):
        assert_uint16_safe(TooLargeTokenizer())
