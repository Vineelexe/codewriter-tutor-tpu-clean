from __future__ import annotations

import argparse
from pathlib import Path

from cw360.config import load_config
from cw360.kaggle import DurableCheckpointStore, verify_checkpoint_bundle
from cw360.model import CodeWriterTutorLM

from scripts.kaggle_tpu_setup_check import build_setup_report
from tests.test_checkpoint_metadata import sample_metadata, tiny_lm_config


def test_kaggle_tpu_scripts_exist() -> None:
    expected = [
        "kaggle_tpu_setup_check.py",
        "kaggle_tpu_train.py",
        "kaggle_tpu_resume_latest.py",
        "kaggle_tpu_save_before_exit.py",
        "kaggle_tpu_handoff.py",
        "kaggle_tpu_verify_checkpoint.py",
        "kaggle_tpu_background_entry.py",
        "kaggle_tpu_quota_note.py",
    ]
    for script in expected:
        assert (Path("scripts") / script).exists()


def test_real_tpu_configs_default_to_background_platform() -> None:
    for config_path in (
        "configs/train_tpu_354m_1024.yaml",
        "configs/train_tpu_420m_1024.yaml",
        "configs/train_tpu_480m_1024.yaml",
        "configs/train_tpu_354m_2048_later.yaml",
    ):
        config = load_config(config_path)
        assert config["platform"] == "kaggle-tpu-background"
        assert config["precision"] == "bf16"
        assert config["drop_last"] is True
        assert config["static_shapes"] is True
        assert config["session_time_limit_minutes"] == 510


def test_setup_check_dry_run_reports_required_fields(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.kaggle_tpu_setup_check._tokenizer_status",
        lambda tokenizer_id, dry_run: {"ok": True, "tokenizer_id": tokenizer_id},
    )
    args = argparse.Namespace(
        config="configs/train_tpu_354m_1024.yaml",
        prepacked_dir=None,
        checkpoint_dir="outputs/checkpoints",
        durable_dir=None,
        dry_run=True,
    )

    report = build_setup_report(args)

    assert report["python_version"]
    assert report["torch_version"]
    assert report["hf_token_present"] is False
    assert report["tokenizer_load_check"]["ok"] is True
    assert report["session_time_limit_minutes"] == 510
    assert "quota" in report["quota_reminder"].lower()


def test_local_durable_store_uploads_checkpoint_bundle(checkpoint_tmp_path, tmp_path) -> None:
    from cw360.checkpoint import save_checkpoint

    model = CodeWriterTutorLM(tiny_lm_config(), init_seed=1)
    checkpoint = save_checkpoint(
        model=model,
        output_dir=checkpoint_tmp_path,
        metadata=sample_metadata(step=4),
    )
    assert checkpoint is not None

    store = DurableCheckpointStore(store_type="local_dir", local_dir=tmp_path / "durable")
    record = store.upload_checkpoint(checkpoint)

    assert record.verified is True
    assert Path(record.durable_uri).exists()
    verify_checkpoint_bundle(checkpoint)
