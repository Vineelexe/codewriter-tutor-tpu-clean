from __future__ import annotations

import os
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from cw360.config import load_model_config
from cw360.tokenizer.special_tokens import DEFAULT_TOKENIZER_ID


class TokenizerLoadError(RuntimeError):
    """Raised when a tokenizer cannot be loaded from Hugging Face or local cache."""


def tokenizer_id_from_config(config_path: str | Path | None) -> str:
    if config_path is None:
        return DEFAULT_TOKENIZER_ID
    return load_model_config(config_path).tokenizer_name


def load_tokenizer(
    tokenizer_id: str | None = None,
    *,
    token: str | None = None,
    use_fast: bool = True,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
) -> PreTrainedTokenizerBase:
    resolved_tokenizer_id = tokenizer_id or DEFAULT_TOKENIZER_ID
    hf_token = token if token is not None else os.environ.get("HF_TOKEN")

    kwargs: dict[str, object] = {
        "use_fast": use_fast,
        "trust_remote_code": trust_remote_code,
    }
    if local_files_only:
        kwargs["local_files_only"] = True
    if hf_token:
        kwargs["token"] = hf_token

    try:
        return AutoTokenizer.from_pretrained(resolved_tokenizer_id, **kwargs)
    except Exception as exc:  # pragma: no cover - exact exception type varies by environment.
        raise TokenizerLoadError(
            "failed to load tokenizer "
            f"{resolved_tokenizer_id!r}. If this is a private or gated Hugging Face "
            "repository, set HF_TOKEN or run `huggingface-cli login`. If this machine "
            "has no network access, pre-populate the Hugging Face cache before running "
            "verification."
        ) from exc


def ensure_pad_token_for_batching(tokenizer: PreTrainedTokenizerBase) -> bool:
    if tokenizer.pad_token is not None:
        return False
    if tokenizer.eos_token is None:
        raise ValueError("tokenizer has no pad token and no eos token to reuse for batching")
    tokenizer.pad_token = tokenizer.eos_token
    return True
