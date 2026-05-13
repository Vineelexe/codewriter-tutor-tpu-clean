from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Literal

import torch


PrecisionName = Literal["float32", "bf16"]
BackendName = Literal["cpu", "xla_tpu"]


@dataclass(frozen=True, slots=True)
class PrecisionConfig:
    precision: PrecisionName = "float32"
    backend: BackendName = "cpu"


def autocast_context(config: PrecisionConfig) -> Any:
    if config.precision == "float32":
        return nullcontext()
    if config.precision == "bf16" and config.backend == "cpu":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    if config.precision == "bf16" and config.backend == "xla_tpu":
        return nullcontext()
    raise ValueError(f"unsupported precision/backend combination: {config}")


def uses_cuda_grad_scaler(config: PrecisionConfig) -> bool:
    return False
