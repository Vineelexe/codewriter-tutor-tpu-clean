from __future__ import annotations

from cw360.data.fim import (
    FALLBACK_FIM_MIDDLE,
    FALLBACK_FIM_PREFIX,
    FALLBACK_FIM_SUFFIX,
    format_fim_code,
)


class FIMTokenizer:
    all_special_tokens = ["<fim_prefix>", "<fim_suffix>", "<fim_middle>"]
    special_tokens_map = {
        "additional_special_tokens": ["<fim_prefix>", "<fim_suffix>", "<fim_middle>"]
    }

    def get_vocab(self) -> dict[str, int]:
        return {"<fim_prefix>": 1, "<fim_suffix>": 2, "<fim_middle>": 3}


def test_fim_uses_tokenizer_supported_markers_and_preserves_parts() -> None:
    code = "def add(a, b):\n    total = a + b\n    return total\n"
    start = code.index("    total")
    end = code.index("    return")

    formatted = format_fim_code(
        code,
        tokenizer=FIMTokenizer(),
        middle_start=start,
        middle_end=end,
    )

    assert formatted.applied is True
    assert formatted.markers.tokenizer_supported is True
    assert formatted.parts.prefix == "def add(a, b):\n"
    assert formatted.parts.middle == "    total = a + b\n"
    assert formatted.parts.suffix == "    return total\n"
    assert formatted.text == (
        "<fim_prefix>def add(a, b):\n"
        "<fim_suffix>    return total\n"
        "<fim_middle>    total = a + b\n"
    )


def test_fim_falls_back_to_documented_plain_text_markers() -> None:
    formatted = format_fim_code("abc", tokenizer=None, middle_start=1, middle_end=2)

    assert formatted.markers.tokenizer_supported is False
    assert formatted.text == f"{FALLBACK_FIM_PREFIX}a{FALLBACK_FIM_SUFFIX}c{FALLBACK_FIM_MIDDLE}b"


def test_fim_probability_can_leave_example_unchanged() -> None:
    formatted = format_fim_code("print('x')\n", fim_probability=0.0)

    assert formatted.applied is False
    assert formatted.text == "print('x')\n"
