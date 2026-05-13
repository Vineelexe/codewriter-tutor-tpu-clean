from __future__ import annotations

import torch
from torch import nn

from cw360.model.rmsnorm import RMSNorm


def initialize_module(module: nn.Module, *, std: float = 0.02) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std)
    elif isinstance(module, RMSNorm):
        nn.init.ones_(module.weight)


def deterministic_init(module: nn.Module, *, seed: int = 1234, std: float = 0.02) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        module.apply(lambda child: initialize_module(child, std=std))
