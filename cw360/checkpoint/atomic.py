from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch


def temporary_path_for(final_path: str | Path) -> Path:
    final = Path(final_path)
    return final.with_name(f".{final.name}.{uuid.uuid4().hex}.tmp")


def atomic_torch_save(
    obj: Any,
    final_path: str | Path,
    *,
    save_fn: Callable[[Any, str], None] | None = None,
) -> Path:
    final = Path(final_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        raise FileExistsError(f"checkpoint already exists: {final}")

    tmp_path = temporary_path_for(final)
    try:
        if save_fn is None:
            torch.save(obj, tmp_path)
        else:
            save_fn(obj, str(tmp_path))
        fsync_file(tmp_path)
        promote_without_overwrite(tmp_path, final)
        fsync_directory_best_effort(final.parent)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return final


def atomic_write_json(data: dict[str, Any], final_path: str | Path) -> Path:
    final = Path(final_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        raise FileExistsError(f"file already exists: {final}")

    tmp_path = temporary_path_for(final)
    try:
        with tmp_path.open("x", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        promote_without_overwrite(tmp_path, final)
        fsync_directory_best_effort(final.parent)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return final


def fsync_file(path: str | Path) -> None:
    try:
        with Path(path).open("rb") as handle:
            os.fsync(handle.fileno())
    except OSError:
        return


def promote_without_overwrite(source: str | Path, final_path: str | Path) -> Path:
    source_path = Path(source)
    final = Path(final_path)
    try:
        os.link(source_path, final)
    except FileExistsError:
        raise FileExistsError(f"destination already exists: {final}") from None
    except OSError:
        if final.exists():
            raise FileExistsError(f"destination already exists: {final}")
        source_path.rename(final)
    else:
        source_path.unlink()
    return final


def fsync_directory_best_effort(path: str | Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    try:
        fd = os.open(Path(path), os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
