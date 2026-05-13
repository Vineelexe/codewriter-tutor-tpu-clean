from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class DatasetSourceConfig:
    name: str
    dataset_id: str | None = None
    subset: str | None = None
    split: str = "train"
    streaming: bool = True
    extractor: str | None = None
    local_path: str | None = None
    gated: bool = False
    max_preview_examples: int = 20
    extra: dict[str, Any] = field(default_factory=dict)


def load_dataset_source_configs(path: str | Path) -> dict[str, DatasetSourceConfig]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"dataset config {config_path} must be a mapping")
    sources = raw.get("sources")
    if not isinstance(sources, dict):
        raise ValueError(f"dataset config {config_path} must contain a sources mapping")
    parsed: dict[str, DatasetSourceConfig] = {}
    for name, value in sources.items():
        if not isinstance(value, dict):
            raise ValueError(f"dataset source {name} must be a mapping")
        known = {
            "dataset_id",
            "subset",
            "split",
            "streaming",
            "extractor",
            "local_path",
            "gated",
            "max_preview_examples",
        }
        parsed[str(name)] = DatasetSourceConfig(
            name=str(name),
            dataset_id=value.get("dataset_id"),
            subset=value.get("subset"),
            split=str(value.get("split", "train")),
            streaming=bool(value.get("streaming", True)),
            extractor=value.get("extractor") or str(name),
            local_path=value.get("local_path"),
            gated=bool(value.get("gated", False)),
            max_preview_examples=int(value.get("max_preview_examples", 20)),
            extra={str(key): item for key, item in value.items() if key not in known},
        )
    return parsed
