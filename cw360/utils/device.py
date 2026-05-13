from __future__ import annotations


def get_local_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def require_torch_xla() -> object:
    try:
        import torch_xla.core.xla_model as xm
    except ImportError as exc:
        raise RuntimeError(
            "torch_xla is required for TPU execution. Install the Kaggle/TPU-compatible "
            "torch-xla build only in TPU environments; local CPU tests do not require it."
        ) from exc
    return xm
