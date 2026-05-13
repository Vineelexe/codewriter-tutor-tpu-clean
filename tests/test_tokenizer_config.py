from __future__ import annotations

import pytest

from cw360.config import ConfigError, parse_config
from cw360.tokenizer.loader import load_tokenizer, tokenizer_id_from_config
from cw360.tokenizer.special_tokens import DEFAULT_TOKENIZER_ID


def test_default_tokenizer_id_is_starcoder2_15b() -> None:
    parsed = parse_config(
        {
            "config_type": "model",
            "size_label": "tiny",
            "max_position_embeddings": 128,
            "hidden_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 256,
        }
    )

    assert parsed.tokenizer_name == DEFAULT_TOKENIZER_ID
    assert DEFAULT_TOKENIZER_ID == "bigcode/starcoder2-15b"


def test_non_starcoder2_tokenizer_config_is_rejected() -> None:
    with pytest.raises(ConfigError, match="StarCoder2"):
        parse_config(
            {
                "config_type": "model",
                "size_label": "tiny",
                "tokenizer_name": "gpt2",
                "max_position_embeddings": 128,
                "hidden_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "intermediate_size": 256,
            }
        )


def test_tokenizer_id_from_missing_config_uses_default() -> None:
    assert tokenizer_id_from_config(None) == DEFAULT_TOKENIZER_ID


def test_loader_uses_hf_token_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class DummyAutoTokenizer:
        @staticmethod
        def from_pretrained(tokenizer_id: str, **kwargs: object) -> object:
            calls.append((tokenizer_id, kwargs))
            return object()

    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    monkeypatch.setattr("cw360.tokenizer.loader.AutoTokenizer", DummyAutoTokenizer)

    tokenizer = load_tokenizer(DEFAULT_TOKENIZER_ID)

    assert tokenizer is not None
    assert calls == [
        (
            DEFAULT_TOKENIZER_ID,
            {"use_fast": True, "trust_remote_code": False, "token": "hf_test_token"},
        )
    ]
