from __future__ import annotations

import pytest
import torch

from cw360.checkpoint.atomic import atomic_torch_save, atomic_write_json, temporary_path_for


def test_atomic_json_write_creates_final_file_without_temp_leftover(checkpoint_tmp_path) -> None:
    path = checkpoint_tmp_path / "metadata.json"

    atomic_write_json({"step": 1}, path)

    assert path.read_text(encoding="utf-8").strip().startswith("{")
    assert list(checkpoint_tmp_path.glob("*.tmp")) == []


def test_atomic_writes_refuse_to_overwrite(checkpoint_tmp_path) -> None:
    path = checkpoint_tmp_path / "checkpoint.pt"
    path.write_bytes(b"existing")

    with pytest.raises(FileExistsError):
        atomic_torch_save({"step": 1}, path)
    with pytest.raises(FileExistsError):
        atomic_write_json({"step": 1}, checkpoint_tmp_path / "checkpoint.pt")


def test_atomic_torch_save_round_trip(checkpoint_tmp_path) -> None:
    path = checkpoint_tmp_path / "checkpoint.pt"

    atomic_torch_save({"tensor": torch.tensor([1, 2, 3])}, path)
    loaded = torch.load(path, weights_only=False)

    assert loaded["tensor"].tolist() == [1, 2, 3]
    assert temporary_path_for(path).name.startswith(".checkpoint.pt.")
