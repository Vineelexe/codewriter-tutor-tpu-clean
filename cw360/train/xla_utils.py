from __future__ import annotations

from typing import Any


class TorchXLANotAvailable(RuntimeError):
    pass


def import_xla_model() -> Any:
    try:
        import torch_xla.core.xla_model as xm
    except ImportError as exc:
        raise TorchXLANotAvailable(
            "torch_xla is required only for TPU training. Install a PyTorch/XLA build "
            "matching the TPU runtime, then run scripts/tpu_train.py inside that TPU environment."
        ) from exc
    return xm


def xla_device() -> Any:
    xm = import_xla_model()
    return xm.xla_device()


def xla_optimizer_step(optimizer: Any) -> None:
    xm = import_xla_model()
    xm.optimizer_step(optimizer)
    xm.mark_step()


def is_xla_main_process() -> bool:
    xm = import_xla_model()
    return int(xm.get_ordinal()) == 0
