from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class AdamWConfig:
    learning_rate: float
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1.0e-8


def build_adamw(
    parameters: Iterable[torch.nn.Parameter],
    config: AdamWConfig,
) -> torch.optim.AdamW:
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if not 0.0 <= config.beta1 < 1.0 or not 0.0 <= config.beta2 < 1.0:
        raise ValueError("AdamW betas must be in [0, 1)")
    if config.eps <= 0:
        raise ValueError("eps must be positive")
    return torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )
