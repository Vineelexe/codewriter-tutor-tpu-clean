from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from cw360.prepack.config import PrepackFactoryConfig
from cw360.prepack.manifest import write_json


def kaggle_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    slug = re.sub(r"-+", "-", slug)
    if not slug:
        raise ValueError("Kaggle dataset slug cannot be empty")
    return slug


def build_kaggle_dataset_metadata(config: PrepackFactoryConfig) -> dict[str, Any]:
    dataset_name = config.prepack.dataset_name
    return {
        "title": dataset_name,
        "id": f"{config.prepack.kaggle_owner}/{kaggle_slug(dataset_name)}",
        "licenses": [{"name": "other"}],
    }


def write_kaggle_dataset_metadata(
    output_dir: str | Path,
    config: PrepackFactoryConfig,
) -> dict[str, Any]:
    metadata = build_kaggle_dataset_metadata(config)
    write_json(Path(output_dir) / "dataset-metadata.json", metadata)
    return metadata


def run_kaggle_dataset_upload(
    output_dir: str | Path,
    *,
    create: bool,
    private: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = ["kaggle", "datasets", "create" if create else "version", "-p", str(output_dir)]
    if create and private:
        command.append("--private")
    if not create:
        command.extend(["-m", "Update prepacked TPU shards"])
    return subprocess.run(command, check=False, capture_output=True, text=True)
