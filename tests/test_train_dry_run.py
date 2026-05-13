from __future__ import annotations

from pathlib import Path

import yaml

from cw360.config import load_config
from cw360.prepack.config import load_prepack_config
from scripts.make_tiny_prepacked_dataset import make_tiny_prepacked_dataset
from scripts.train_dry_run import build_dry_run_summary

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_train_dry_run_loads_manifest_and_inspects_fixed_tiny_batch(tmp_path: Path) -> None:
    prepacked_dir = tmp_path / "prepacked"
    prepack_config = load_prepack_config(CONFIG_DIR / "prepack_tpu_1024.yaml")
    make_tiny_prepacked_dataset(
        config=prepack_config,
        output_dir=prepacked_dir,
        num_sequences=8,
    )
    config = load_config(CONFIG_DIR / "train_tpu_354m_1024.yaml")
    config.update(
        {
            "model_config_path": str(CONFIG_DIR / "model_354m.yaml"),
            "prepacked_shards_path": str(prepacked_dir),
            "prepacked_dataset_path": str(prepacked_dir),
            "prepacked_manifest_path": str(prepacked_dir / "manifest.json"),
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "micro_batch_size": 4,
            "fixed_batch_size": 4,
        }
    )
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    summary = build_dry_run_summary(config_path)

    assert summary["ok"] is True
    assert summary["manifest"]["available"] is True
    assert summary["batch"]["available"] is True
    assert summary["batch"]["input_ids_shape"] == [4, 1024]
    assert summary["batch"]["labels_shape"] == [4, 1024]
    assert summary["checkpoint_path"]["writable"] is True
    assert summary["time_guards"]["save_before_exit_minutes"] == 20


def test_train_dry_run_no_tpu_required_allows_missing_local_kaggle_manifest() -> None:
    summary = build_dry_run_summary(
        CONFIG_DIR / "train_tpu_354m_1024.yaml",
        no_tpu_required=True,
    )

    assert summary["ok"] is True
    assert summary["manifest"]["available"] is False
    assert summary["batch"]["available"] is False
    assert summary["checkpoint_path"]["writable"] is None
