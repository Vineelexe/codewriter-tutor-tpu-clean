from __future__ import annotations

import json
from pathlib import Path

import yaml

from cw360.checkpoint import save_checkpoint
from cw360.local.export import export_checkpoint
from cw360.model import CodeWriterTutorLM
from tests.test_checkpoint_metadata import sample_cursor, sample_metadata, tiny_lm_config


def _write_tiny_checkpoint(root: Path, *, with_cursor: bool = True) -> Path:
    config = tiny_lm_config(tokenizer_name="bigcode/starcoder2-15b")
    model = CodeWriterTutorLM(config, init_seed=17)
    metadata = sample_metadata(
        backend="xla_tpu" if with_cursor else "cpu",
        platform="kaggle-tpu-background" if with_cursor else "local-cpu-test",
        tpu_data_cursor=sample_cursor() if with_cursor else None,
    )
    checkpoint = save_checkpoint(model=model, output_dir=root, metadata=metadata)
    assert checkpoint is not None
    return checkpoint


def test_export_checkpoint_writes_required_layout(tmp_path: Path) -> None:
    checkpoint = _write_tiny_checkpoint(tmp_path / "checkpoints")

    result = export_checkpoint(
        checkpoint_path=checkpoint,
        output_dir=tmp_path / "exported_model",
        allow_test_tokenizer=True,
    )

    expected_files = {
        "config.yaml",
        "model.safetensors",
        "tokenizer_ref.txt",
        "generation_config.yaml",
        "metadata.json",
        "run_local.py",
        "quantization_report.json",
        "demo_prompts.json",
    }
    assert {path.name for path in result.output_dir.iterdir()} == expected_files
    assert result.tokenizer_ref_path.read_text(encoding="utf-8").strip() == (
        "bigcode/starcoder2-15b"
    )
    assert yaml.safe_load(result.config_path.read_text(encoding="utf-8"))["size_label"] == "tiny"


def test_export_metadata_preserves_data_lineage_and_cursor(tmp_path: Path) -> None:
    checkpoint = _write_tiny_checkpoint(tmp_path / "checkpoints")

    result = export_checkpoint(
        checkpoint_path=checkpoint,
        output_dir=tmp_path / "exported_model",
        allow_test_tokenizer=True,
    )
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert metadata["runtime_target"] == "pytorch_fp32_cpu"
    assert metadata["data_snapshot"]["prepacked_manifest_hash"] == "prepacked-hash"
    assert metadata["data_snapshot"]["seq_len_plus_one"] == 33
    assert metadata["tpu_data_cursor"]["epoch"] == 1
    assert metadata["tpu_data_cursor"]["shard_order_seed"] == 123
    assert metadata["tpu_data_cursor"]["shard_index"] == 2
    assert metadata["tpu_data_cursor"]["sequence_offset"] == 7
    assert metadata["tpu_data_cursor"]["tokens_seen"] == 320


def test_quantization_report_is_honest_about_unsupported_paths(tmp_path: Path) -> None:
    checkpoint = _write_tiny_checkpoint(tmp_path / "checkpoints", with_cursor=False)

    result = export_checkpoint(
        checkpoint_path=checkpoint,
        output_dir=tmp_path / "exported_model",
        allow_test_tokenizer=True,
    )
    report = json.loads(result.quantization_report_path.read_text(encoding="utf-8"))

    assert report["pytorch_dynamic_int8"]["status"] == "available_as_runtime_attempt"
    assert report["bitsandbytes"]["status"] == "not_enabled_for_cpu_runtime"
    assert report["custom_4bit"]["status"] == "not_implemented"
    assert report["gguf"]["status"] == "unsupported"
