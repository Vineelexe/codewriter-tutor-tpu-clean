from __future__ import annotations

from pathlib import Path

import torch
from transformers import PreTrainedTokenizerBase

from cw360.checkpoint.manager import load_checkpoint
from cw360.config import load_model_config
from cw360.model.lm import CodeWriterTutorLM
from cw360.tokenizer.loader import ensure_pad_token_for_batching, load_tokenizer

def load_inference_model(
    config_path: str | Path,
    checkpoint_path: str | Path | None = None,
    *,
    device: str | torch.device = "cpu",
    strict: bool = True,
    random_weights: bool = False,
) -> tuple[CodeWriterTutorLM, PreTrainedTokenizerBase]:
    config = load_model_config(config_path)
    model = CodeWriterTutorLM(config)
    
    if checkpoint_path is not None and not random_weights:
        load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            expected_model_size_label=config.size_label,
            strict=strict,
            map_location=device,
        )
    
    model.to(device)
    model.eval()
    
    tokenizer = load_tokenizer(config.tokenizer_name)
    ensure_pad_token_for_batching(tokenizer)
    
    return model, tokenizer
