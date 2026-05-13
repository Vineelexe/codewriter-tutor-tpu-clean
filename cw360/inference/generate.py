from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase

from cw360.model.lm import CodeWriterTutorLM

@torch.inference_mode()
def generate_text(
    model: CodeWriterTutorLM,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    use_cache: bool = True,
    device: str | torch.device = "cpu",
) -> str:
    if seed is not None:
        torch.manual_seed(seed)
        
    model.to(device)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    eos_token_id = tokenizer.eos_token_id
    
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=eos_token_id,
        use_cache=use_cache,
        attention_mask=attention_mask,
    )
    
    prompt_len = input_ids.shape[1]
    generated_ids = output_ids[0, prompt_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)
