from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from cw360.checkpoint.metadata import CheckpointMetadata


CHECKPOINT_FILENAME_RE = re.compile(
    r"^cw360_(?P<model_size>[A-Za-z0-9]+)_step(?P<step>\d{8})_"
    r"loss(?P<loss>\d+\.\d{3})_stage_(?P<training_stage>.+)_train_"
    r"(?P<trainer>[^_]+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<platform>.+)\.pt$"
)


@dataclass(frozen=True, slots=True)
class CheckpointNameParts:
    model_size_label: str
    step: int
    train_loss: float
    training_stage: str
    trainer: str
    date: str
    platform: str


def checkpoint_filename(metadata: CheckpointMetadata) -> str:
    for field_name, value in {
        "model_size_label": metadata.model_size_label,
        "training_stage": metadata.training_stage,
        "trained_by": metadata.trained_by,
        "date": metadata.date,
        "platform": metadata.platform,
    }.items():
        if "/" in value or "\\" in value:
            raise ValueError(f"{field_name} must not contain path separators")
    return (
        f"cw360_{metadata.model_size_label}_step{metadata.step:08d}_"
        f"loss{metadata.train_loss:.3f}_stage_{metadata.training_stage}_train_"
        f"{metadata.trained_by}_{metadata.date}_{metadata.platform}.pt"
    )


def checkpoint_path(directory: str | Path, metadata: CheckpointMetadata) -> Path:
    return Path(directory) / checkpoint_filename(metadata)


def parse_checkpoint_filename(filename: str | Path) -> CheckpointNameParts:
    name = Path(filename).name
    match = CHECKPOINT_FILENAME_RE.match(name)
    if match is None:
        raise ValueError(f"invalid checkpoint filename: {name}")
    groups = match.groupdict()
    return CheckpointNameParts(
        model_size_label=groups["model_size"],
        step=int(groups["step"]),
        train_loss=float(groups["loss"]),
        training_stage=groups["training_stage"],
        trainer=groups["trainer"],
        date=groups["date"],
        platform=groups["platform"],
    )


def is_checkpoint_filename(filename: str | Path) -> bool:
    return CHECKPOINT_FILENAME_RE.match(Path(filename).name) is not None
