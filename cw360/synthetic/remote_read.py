from __future__ import annotations

import shutil
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from cw360.synthetic.local_writer import write_rows_jsonl

DatasetLoader = Callable[..., Iterable[Mapping[str, Any]]]


class SyntheticRemoteReadError(RuntimeError):
    pass


def iter_local_jsonl_rows(paths: Iterable[str | Path]) -> Iterator[dict[str, Any]]:
    import json

    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise SyntheticRemoteReadError(f"{path} contains a non-object JSON row")
                yield payload


def iter_hf_dataset_rows(
    repo_id: str,
    *,
    split: str = "train",
    streaming: bool = True,
    dataset_loader: DatasetLoader | None = None,
    token: str | None = None,
) -> Iterator[dict[str, Any]]:
    loader = dataset_loader or _load_dataset
    try:
        dataset = loader(repo_id, split=split, streaming=streaming, token=token)
    except Exception as exc:
        message = f"could not read Hugging Face dataset {repo_id}: {exc}"
        raise SyntheticRemoteReadError(message) from exc
    for row in dataset:
        yield dict(row)


def pull_synthetic_rows_to_dir(
    rows: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    shard_size: int = 1000,
    overwrite: bool = False,
) -> list[Path]:
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    target = Path(output_dir)
    if target.exists() and overwrite:
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    buffer: list[dict[str, Any]] = []
    shard_index = 0
    for row in rows:
        buffer.append(dict(row))
        if len(buffer) >= shard_size:
            path = target / f"structured_{shard_index:05d}.jsonl"
            write_rows_jsonl(buffer, path)
            written.append(path)
            buffer = []
            shard_index += 1
    if buffer:
        path = target / f"structured_{shard_index:05d}.jsonl"
        write_rows_jsonl(buffer, path)
        written.append(path)
    return written


def _load_dataset(*args: Any, **kwargs: Any) -> Iterable[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SyntheticRemoteReadError("install datasets to read remote synthetic shards") from exc
    return load_dataset(*args, **kwargs)
