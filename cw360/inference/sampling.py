from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class SamplingConfig:
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_k: int | None = None
    top_p: float | None = None
    seed: int | None = None
    use_cache: bool = True
