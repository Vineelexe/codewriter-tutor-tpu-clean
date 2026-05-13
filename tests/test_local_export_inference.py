from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from cw360.checkpoint import save_checkpoint
from cw360.local.export import export_checkpoint
from cw360.model import CodeWriterTutorLM
from tests.test_checkpoint_metadata import sample_metadata, tiny_lm_config

ROOT = Path(__file__).resolve().parents[1]


def _export_tiny_model(tmp_path: Path) -> Path:
    config = tiny_lm_config(tokenizer_name="bigcode/starcoder2-15b")
    model = CodeWriterTutorLM(config, init_seed=23)
    checkpoint = save_checkpoint(
        model=model,
        output_dir=tmp_path / "checkpoints",
        metadata=sample_metadata(),
    )
    assert checkpoint is not None
    result = export_checkpoint(
        checkpoint_path=checkpoint,
        output_dir=tmp_path / "exported_model",
        generation_config={"max_new_tokens": 2, "temperature": 0.0, "seed": 7},
        allow_test_tokenizer=True,
    )
    return result.output_dir


def test_exported_run_local_generates_with_test_tokenizer(tmp_path: Path) -> None:
    exported = _export_tiny_model(tmp_path)
    output_path = exported / "single_prompt_output.json"

    completed = _run_exported(
        exported,
        "--prompt",
        "Write a Python function to add two numbers.",
        "--max-new-tokens",
        "2",
        "--test-tokenizer",
        "--save-output",
        str(output_path),
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload[0]["name"] == "prompt"
    assert payload[0]["prompt"].startswith("Write a Python function")
    assert isinstance(payload[0]["output"], str)
    assert payload[0]["quantization"] == "none"


def test_exported_run_local_demo_saves_fixed_prompt_outputs(tmp_path: Path) -> None:
    exported = _export_tiny_model(tmp_path)
    output_path = exported / "demo_outputs.json"

    completed = _run_exported(
        exported,
        "--demo",
        "--max-new-tokens",
        "1",
        "--test-tokenizer",
        "--save-output",
        str(output_path),
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert {item["name"] for item in payload} == {
        "code_writing",
        "debugging",
        "explanation",
        "tests",
    }


def test_exported_run_local_dynamic_int8_attempt_is_explicit(tmp_path: Path) -> None:
    exported = _export_tiny_model(tmp_path)
    output_path = exported / "int8_output.json"

    completed = _run_exported(
        exported,
        "--prompt",
        "Explain a Python ValueError.",
        "--max-new-tokens",
        "1",
        "--test-tokenizer",
        "--quantization",
        "dynamic-int8",
        "--save-output",
        str(output_path),
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload[0]["quantization"] == "dynamic-int8"


def _run_exported(exported: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, str(exported / "run_local.py"), *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
