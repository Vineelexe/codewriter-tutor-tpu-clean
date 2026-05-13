from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cw360.checkpoint.metadata import metadata_path_for_checkpoint


@dataclass(frozen=True, slots=True)
class ExportLayout:
    model_config_path: Path
    checkpoint_path: Path
    metadata_path: Path
    tokenizer_name: str
    safetensors_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_config_path": str(self.model_config_path),
            "checkpoint_path": str(self.checkpoint_path),
            "metadata_path": str(self.metadata_path),
            "tokenizer_name": self.tokenizer_name,
            "safetensors_path": (
                None if self.safetensors_path is None else str(self.safetensors_path)
            ),
            "gguf_supported": False,
            "runtime_target": "pytorch_cpu_inference",
        }


def validate_export_layout(
    *,
    model_config_path: str | Path,
    checkpoint_path: str | Path,
    tokenizer_name: str,
    safetensors_path: str | Path | None = None,
    require_files: bool = True,
) -> ExportLayout:
    config = Path(model_config_path)
    checkpoint = Path(checkpoint_path)
    metadata = metadata_path_for_checkpoint(checkpoint)
    safetensors = None if safetensors_path is None else Path(safetensors_path)
    if not tokenizer_name:
        raise ValueError("tokenizer_name must be provided for local CPU inference exports")
    if require_files:
        _require_file(config, "model config")
        _require_file(checkpoint, "checkpoint")
        _require_file(metadata, "checkpoint metadata")
        if safetensors is not None:
            _require_file(safetensors, "safetensors export")
    return ExportLayout(
        model_config_path=config,
        checkpoint_path=checkpoint,
        metadata_path=metadata,
        tokenizer_name=tokenizer_name,
        safetensors_path=safetensors,
    )


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
