from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cw360.checkpoint import metadata_path_for_checkpoint, read_metadata_json
from cw360.checkpoint.atomic import atomic_write_json
from cw360.checkpoint.validate import validate_metadata
from cw360.utils.hashing import sha256_file


@dataclass(frozen=True, slots=True)
class CheckpointUploadRecord:
    checkpoint_path: str
    metadata_path: str
    checkpoint_sha256: str
    metadata_sha256: str
    durable_uri: str
    verified: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DurableCheckpointStore:
    def __init__(
        self,
        *,
        store_type: str,
        repo_id: str | None = None,
        local_dir: str | Path | None = None,
        path_prefix: str = "checkpoints",
        private: bool = True,
        token_env: str = "HF_TOKEN",
    ) -> None:
        self.store_type = store_type
        self.repo_id = repo_id
        self.local_dir = None if local_dir is None else Path(local_dir)
        self.path_prefix = path_prefix.strip("/")
        self.private = private
        self.token_env = token_env

    def status(self) -> dict[str, Any]:
        if self.store_type == "hf_model_repo":
            return {
                "type": self.store_type,
                "repo_id_configured": bool(self.repo_id),
                "token_present": bool(os.environ.get(self.token_env)),
                "path_prefix": self.path_prefix,
            }
        if self.store_type == "local_dir":
            return {
                "type": self.store_type,
                "path": None if self.local_dir is None else str(self.local_dir),
                "path_exists": bool(self.local_dir and self.local_dir.exists()),
            }
        return {"type": self.store_type, "supported": False}

    def upload_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        dry_run: bool = False,
    ) -> CheckpointUploadRecord:
        checkpoint = Path(checkpoint_path)
        metadata_path = metadata_path_for_checkpoint(checkpoint)
        verify_checkpoint_bundle(checkpoint)
        record = CheckpointUploadRecord(
            checkpoint_path=str(checkpoint),
            metadata_path=str(metadata_path),
            checkpoint_sha256=sha256_file(checkpoint),
            metadata_sha256=sha256_file(metadata_path),
            durable_uri=self._durable_uri(checkpoint),
            verified=False,
        )
        if dry_run:
            return record
        if self.store_type == "local_dir":
            return self._upload_local(checkpoint, record)
        if self.store_type == "hf_model_repo":
            return self._upload_hf(checkpoint, record)
        raise ValueError(f"unsupported durable checkpoint store type: {self.store_type}")

    def _upload_local(self, checkpoint: Path, record: CheckpointUploadRecord) -> CheckpointUploadRecord:
        if self.local_dir is None:
            raise ValueError("local durable checkpoint store requires local_dir")
        target_dir = self.local_dir / self.path_prefix
        target_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = metadata_path_for_checkpoint(checkpoint)
        target_checkpoint = target_dir / checkpoint.name
        target_metadata = target_dir / metadata_path.name
        _copy_without_overwrite(checkpoint, target_checkpoint)
        _copy_without_overwrite(metadata_path, target_metadata)
        verified = (
            sha256_file(target_checkpoint) == record.checkpoint_sha256
            and sha256_file(target_metadata) == record.metadata_sha256
        )
        upload_record = CheckpointUploadRecord(
            **{**record.to_dict(), "durable_uri": str(target_checkpoint), "verified": verified}
        )
        _write_record_once(upload_record, target_checkpoint.with_suffix(".upload.json"))
        return upload_record

    def _upload_hf(self, checkpoint: Path, record: CheckpointUploadRecord) -> CheckpointUploadRecord:
        if not self.repo_id:
            raise ValueError("hf_model_repo durable store requires repo_id")
        token = os.environ.get(self.token_env)
        if not token:
            raise ValueError(f"set {self.token_env} before uploading checkpoints to Hugging Face")
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:  # pragma: no cover - depends on Kaggle environment.
            raise RuntimeError("huggingface_hub is required for hf_model_repo uploads") from exc

        api = HfApi(token=token)
        api.create_repo(repo_id=self.repo_id, repo_type="model", private=self.private, exist_ok=True)
        metadata_path = metadata_path_for_checkpoint(checkpoint)
        checksum_path = checkpoint.with_suffix(".upload.json")
        checksum_path.write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        try:
            for path in (checkpoint, metadata_path, checksum_path):
                api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=self._path_in_repo(path.name),
                    repo_id=self.repo_id,
                    repo_type="model",
                )
        finally:
            checksum_path.unlink(missing_ok=True)
        return CheckpointUploadRecord(**{**record.to_dict(), "verified": True})

    def _durable_uri(self, checkpoint: Path) -> str:
        if self.store_type == "hf_model_repo":
            repo = self.repo_id or "<unset-repo>"
            return f"hf://{repo}/{self._path_in_repo(checkpoint.name)}"
        if self.store_type == "local_dir" and self.local_dir is not None:
            return str(self.local_dir / self.path_prefix / checkpoint.name)
        return f"{self.store_type}:{checkpoint.name}"

    def _path_in_repo(self, filename: str) -> str:
        return f"{self.path_prefix}/{filename}" if self.path_prefix else filename


def build_durable_store_from_config(
    config: dict[str, Any],
    *,
    durable_dir: str | Path | None = None,
) -> DurableCheckpointStore:
    raw = config.get("durable_checkpoint_store") or {}
    if durable_dir is not None:
        return DurableCheckpointStore(store_type="local_dir", local_dir=durable_dir)
    if not isinstance(raw, dict) or not raw:
        return DurableCheckpointStore(store_type="hf_model_repo", repo_id=os.environ.get("CW360_HF_REPO_ID"))
    return DurableCheckpointStore(
        store_type=str(raw.get("type", "hf_model_repo")),
        repo_id=_none_or_str(raw.get("repo_id") or os.environ.get("CW360_HF_REPO_ID")),
        local_dir=raw.get("path"),
        path_prefix=str(raw.get("path_prefix", "checkpoints")),
        private=bool(raw.get("private", True)),
        token_env=str(raw.get("token_env", "HF_TOKEN")),
    )


def verify_checkpoint_bundle(checkpoint_path: str | Path) -> None:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    metadata_path = metadata_path_for_checkpoint(checkpoint)
    if not metadata_path.exists():
        raise FileNotFoundError(f"checkpoint metadata not found: {metadata_path}")
    validate_metadata(read_metadata_json(metadata_path))


def _copy_without_overwrite(source: Path, destination: Path) -> None:
    if destination.exists():
        if sha256_file(source) == sha256_file(destination):
            return
        raise FileExistsError(f"durable checkpoint destination exists with different content: {destination}")
    tmp = destination.with_name(f".{destination.name}.tmp")
    shutil.copy2(source, tmp)
    tmp.replace(destination)


def _write_record_once(record: CheckpointUploadRecord, destination: Path) -> None:
    payload = record.to_dict()
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing == payload:
            return
        raise FileExistsError(f"durable upload record exists with different content: {destination}")
    atomic_write_json(payload, destination)


def _none_or_str(value: object) -> str | None:
    return None if value is None else str(value)
