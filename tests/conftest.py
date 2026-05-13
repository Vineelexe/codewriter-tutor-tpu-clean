from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest) -> Iterator[Path]:
    base = Path(".pytest-tmp-phase16-fixtures").resolve()
    safe_name = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in request.node.name
    )[:80]
    path = base / f"{safe_name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    yield path


@pytest.fixture
def checkpoint_tmp_path(tmp_path: Path) -> Iterator[Path]:
    yield tmp_path
