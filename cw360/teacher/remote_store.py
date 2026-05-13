from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class UploadResult:
    uploaded: bool
    destination: str
    files: list[str] = field(default_factory=list)


class MockRemoteStore:
    def __init__(self) -> None:
        self.uploads: list[UploadResult] = []

    def upload_dir(
        self,
        local_dir: str | Path,
        *,
        delete_local_after_upload: bool = False,
    ) -> UploadResult:
        files = [
            str(path.relative_to(local_dir))
            for path in Path(local_dir).rglob("*")
            if path.is_file()
        ]
        result = UploadResult(uploaded=True, destination="mock://remote-store", files=sorted(files))
        self.uploads.append(result)
        if delete_local_after_upload:
            shutil.rmtree(local_dir)
        return result


class HuggingFaceDatasetStore:
    def __init__(self, repo_id: str, *, token_env: str = "HF_TOKEN", private: bool = True) -> None:
        self.repo_id = repo_id
        self.token_env = token_env
        self.private = private

    def validate_environment(self) -> None:
        if not os.environ.get(self.token_env):
            raise RuntimeError(f"{self.token_env} is required to upload synthetic shards")

    def upload_dir(
        self,
        local_dir: str | Path,
        *,
        delete_local_after_upload: bool = False,
    ) -> UploadResult:
        self.validate_environment()
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ[self.token_env])
        api.create_repo(
            repo_id=self.repo_id,
            repo_type="dataset",
            private=self.private,
            exist_ok=True,
        )
        api.upload_folder(
            folder_path=str(local_dir),
            repo_id=self.repo_id,
            repo_type="dataset",
            path_in_repo=".",
        )
        files = [
            str(path.relative_to(local_dir))
            for path in Path(local_dir).rglob("*")
            if path.is_file()
        ]
        if delete_local_after_upload:
            shutil.rmtree(local_dir)
        return UploadResult(
            uploaded=True,
            destination=f"hf://datasets/{self.repo_id}",
            files=sorted(files),
        )
