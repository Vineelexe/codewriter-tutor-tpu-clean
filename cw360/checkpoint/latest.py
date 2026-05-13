from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

from cw360.checkpoint.metadata import CheckpointMetadata, metadata_path_for_checkpoint, read_metadata_json
from cw360.checkpoint.naming import is_checkpoint_filename
from cw360.checkpoint.validate import CheckpointValidationError, validate_metadata


@dataclass(frozen=True, slots=True)
class LatestCheckpoint:
    path: Path
    metadata: CheckpointMetadata


def find_latest_checkpoint(directory: str | Path) -> LatestCheckpoint | None:
    best: LatestCheckpoint | None = None
    for path in Path(directory).glob("*.pt"):
        if path.name.endswith(".tmp") or not is_checkpoint_filename(path):
            continue
        metadata_path = metadata_path_for_checkpoint(path)
        if not metadata_path.exists():
            warnings.warn(f"checkpoint without matching metadata ignored: {path}", stacklevel=2)
            continue
        try:
            metadata = read_metadata_json(metadata_path)
            validate_metadata(metadata)
        except (OSError, ValueError, KeyError, CheckpointValidationError) as exc:
            warnings.warn(f"invalid checkpoint metadata ignored for {path}: {exc}", stacklevel=2)
            continue
        if best is None or metadata.step > best.metadata.step:
            best = LatestCheckpoint(path=path, metadata=metadata)
    return best
