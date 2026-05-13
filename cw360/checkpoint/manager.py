from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch

from cw360.checkpoint.atomic import atomic_torch_save
from cw360.checkpoint.metadata import (
    CheckpointMetadata,
    metadata_path_for_checkpoint,
    read_metadata_json,
    write_metadata_json,
)
from cw360.checkpoint.naming import checkpoint_path
from cw360.checkpoint.validate import (
    validate_metadata,
    validate_model_compatibility,
    validate_resume_request,
)


def save_checkpoint(
    *,
    model: torch.nn.Module,
    output_dir: str | Path,
    metadata: CheckpointMetadata,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    config: dict[str, Any] | None = None,
) -> Path | None:
    validate_metadata(metadata)
    final_path = checkpoint_path(output_dir, metadata)
    metadata_path = metadata_path_for_checkpoint(final_path)
    if final_path.exists():
        raise FileExistsError(f"checkpoint already exists: {final_path}")
    if metadata_path.exists():
        raise FileExistsError(f"checkpoint metadata already exists: {metadata_path}")

    if metadata.backend == "xla_tpu" and not is_xla_main_process():
        return None

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "step": metadata.step,
        "tokens_seen": metadata.tokens_seen,
        "sequences_seen": metadata.sequences_seen,
        "config": config if config is not None else extract_model_config(model),
        "metadata": metadata.to_dict(),
        "rng_states": collect_rng_states(),
    }
    atomic_torch_save(payload, final_path, save_fn=xla_save_fn() if metadata.backend == "xla_tpu" else None)
    write_metadata_json(metadata, final_path)
    return final_path


def load_checkpoint(
    *,
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    expected_model_size_label: str | None = None,
    expected_training_stage: str | None = None,
    expected_data_snapshot: Any | None = None,
    allow_training_stage_mismatch: bool = False,
    allow_prepacked_dataset_mismatch: bool = False,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> tuple[CheckpointMetadata, Any | None]:
    path = Path(checkpoint_path)
    payload = torch_load(path, map_location=map_location)
    metadata_file = metadata_path_for_checkpoint(path)
    metadata = (
        read_metadata_json(metadata_file)
        if metadata_file.exists()
        else CheckpointMetadata.from_dict(payload["metadata"])
    )
    validate_metadata(metadata)
    validate_model_compatibility(model, metadata, payload.get("config", {}))
    validate_resume_request(
        metadata,
        expected_model_size_label=expected_model_size_label,
        expected_training_stage=expected_training_stage,
        expected_data_snapshot=expected_data_snapshot,
        allow_training_stage_mismatch=allow_training_stage_mismatch,
        allow_prepacked_dataset_mismatch=allow_prepacked_dataset_mismatch,
    )

    model.load_state_dict(payload["model_state_dict"], strict=strict)
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    return metadata, metadata.tpu_data_cursor


def extract_model_config(model: torch.nn.Module) -> dict[str, Any]:
    config = getattr(model, "config", None)
    if config is None:
        return {}
    if hasattr(config, "model_dump"):
        return dict(config.model_dump())
    return dict(vars(config))


def collect_rng_states() -> dict[str, Any]:
    states: dict[str, Any] = {
        "python_random_state": random.getstate(),
        "torch_cpu_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np
    except ImportError:
        return states
    states["numpy_random_state"] = np.random.get_state()
    return states


def torch_load(path: Path, *, map_location: str | torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def is_xla_main_process() -> bool:
    try:
        import torch_xla.core.xla_model as xm
    except ImportError:
        return True
    return int(xm.get_ordinal()) == 0


def xla_save_fn() -> Any | None:
    try:
        import torch_xla.core.xla_model as xm
    except ImportError:
        return None
    return lambda obj, path: xm.save(obj, path)
