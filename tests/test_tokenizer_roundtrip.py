from __future__ import annotations

from cw360.tokenizer.special_tokens import DEFAULT_TOKENIZER_ID, detect_fim_tokens
from cw360.tokenizer.validate import validate_loaded_tokenizer


class FakeTokenizer:
    eos_token = "<|endoftext|>"
    bos_token = None
    pad_token = None
    pad_token_id = None
    vocab_size = 8
    all_special_tokens = ["<|endoftext|>", "<fim_prefix>", "<fim_middle>", "<fim_suffix>"]
    special_tokens_map = {
        "eos_token": "<|endoftext|>",
        "additional_special_tokens": ["<fim_prefix>", "<fim_middle>", "<fim_suffix>"],
    }

    def __init__(self) -> None:
        self._texts: dict[int, str] = {}

    def get_vocab(self) -> dict[str, int]:
        return {
            "<|endoftext|>": 0,
            "<fim_prefix>": 1,
            "<fim_middle>": 2,
            "<fim_suffix>": 3,
            "def": 4,
            "return": 5,
            "    ": 6,
            "==": 7,
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        token_id = len(self._texts) + 4
        self._texts[token_id] = text
        return [token_id]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        assert skip_special_tokens is False
        return "".join(self._texts[token_id] for token_id in token_ids)


def test_validate_loaded_tokenizer_roundtrips_python_samples() -> None:
    tokenizer = FakeTokenizer()

    report = validate_loaded_tokenizer(
        tokenizer,
        DEFAULT_TOKENIZER_ID,
        require_pad_for_batching=True,
    )

    assert report.vocab_size == 8
    assert report.max_token_id == 7
    assert report.pad_token == "<|endoftext|>"
    assert report.pad_token_was_set_for_batching is True
    assert {check.name for check in report.roundtrip_checks} == {
        "python_code",
        "docstring_heavy_python_code",
    }
    assert all(check.indentation_survives for check in report.roundtrip_checks)
    assert all(check.operators_survive for check in report.roundtrip_checks)


def test_detects_fim_style_special_tokens() -> None:
    assert detect_fim_tokens(FakeTokenizer()) == [
        "<fim_middle>",
        "<fim_prefix>",
        "<fim_suffix>",
    ]
