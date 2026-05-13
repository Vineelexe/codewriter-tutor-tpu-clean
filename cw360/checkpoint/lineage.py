from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cw360.checkpoint.atomic import atomic_write_json
from cw360.checkpoint.metadata import metadata_path_for_checkpoint, read_metadata_json
from cw360.utils.hashing import sha256_file


@dataclass(frozen=True, slots=True)
class HandoffManifest:
    latest_checkpoint_path: str
    checkpoint_sha256: str
    trainer_leaving: str
    next_trainer_expected: str
    prepacked_dataset_identity: dict[str, Any]
    tpu_data_cursor: dict[str, Any] | None
    exact_resume_command: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_handoff_manifest(
    *,
    checkpoint_path: str | Path,
    trainer_leaving: str,
    next_trainer_expected: str,
    exact_resume_command: str,
) -> HandoffManifest:
    path = Path(checkpoint_path)
    metadata = read_metadata_json(metadata_path_for_checkpoint(path))
    snapshot = metadata.data_snapshot
    return HandoffManifest(
        latest_checkpoint_path=str(path),
        checkpoint_sha256=sha256_file(path),
        trainer_leaving=trainer_leaving,
        next_trainer_expected=next_trainer_expected,
        prepacked_dataset_identity={
            "prepacked_dataset_name": snapshot.prepacked_dataset_name,
            "prepacked_dataset_version": snapshot.prepacked_dataset_version,
            "prepacked_manifest_hash": snapshot.prepacked_manifest_hash,
            "shard_manifest_hash": snapshot.shard_manifest_hash,
            "seq_len_plus_one": snapshot.seq_len_plus_one,
            "token_dtype": snapshot.token_dtype,
        },
        tpu_data_cursor=(
            None if metadata.tpu_data_cursor is None else metadata.tpu_data_cursor.to_dict()
        ),
        exact_resume_command=exact_resume_command,
    )


def write_handoff_manifest(
    *,
    checkpoint_path: str | Path,
    trainer_leaving: str,
    next_trainer_expected: str,
    exact_resume_command: str,
    output_path: str | Path | None = None,
) -> Path:
    manifest = build_handoff_manifest(
        checkpoint_path=checkpoint_path,
        trainer_leaving=trainer_leaving,
        next_trainer_expected=next_trainer_expected,
        exact_resume_command=exact_resume_command,
    )
    checkpoint = Path(checkpoint_path)
    target = (
        Path(output_path)
        if output_path is not None
        else checkpoint.with_name(f"{checkpoint.stem}_handoff.json")
    )
    atomic_write_json(manifest.to_dict(), target)
    return target
