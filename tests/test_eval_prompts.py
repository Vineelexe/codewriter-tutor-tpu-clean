from __future__ import annotations

import json
import shutil
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from cw360.eval.prompts import load_eval_prompts


def test_load_fixed_eval_prompts() -> None:
    prompts = load_eval_prompts(
        [Path("eval_prompts/core_prompts.jsonl"), Path("eval_prompts/extended_prompts.jsonl")]
    )

    prompt_ids = {prompt.prompt_id for prompt in prompts}
    assert len(prompts) == 15
    assert "core_average_zero_division" in prompt_ids
    assert "extended_mutable_default_arguments" in prompt_ids


def test_load_eval_prompts_rejects_duplicate_ids(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompts.jsonl"
    record = {"prompt_id": "duplicate", "category": "python", "prompt": "Write Python."}
    prompt_path.write_text(
        json.dumps(record) + "\n" + json.dumps(record) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate eval prompt_id"):
        load_eval_prompts([prompt_path])
