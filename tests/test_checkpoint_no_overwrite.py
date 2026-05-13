from __future__ import annotations

import pytest

from cw360.checkpoint import save_checkpoint
from cw360.model import CodeWriterTutorLM

from tests.test_checkpoint_metadata import sample_metadata, tiny_lm_config


def test_checkpoint_save_never_overwrites_existing_checkpoint(checkpoint_tmp_path) -> None:
    model = CodeWriterTutorLM(tiny_lm_config(), init_seed=1)
    metadata = sample_metadata()

    first = save_checkpoint(model=model, output_dir=checkpoint_tmp_path, metadata=metadata)
    assert first is not None
    with pytest.raises(FileExistsError):
        save_checkpoint(model=model, output_dir=checkpoint_tmp_path, metadata=metadata)


def test_checkpoint_save_rejects_existing_metadata_path(checkpoint_tmp_path) -> None:
    model = CodeWriterTutorLM(tiny_lm_config(), init_seed=1)
    metadata = sample_metadata(step=6)
    expected_checkpoint = checkpoint_tmp_path / (
        "cw360_tiny_step00000006_loss4.231_stage_base_pretrain_train_vineel_"
        "2026-05-23_local-cpu-test.pt"
    )
    expected_checkpoint.with_suffix(".json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="metadata"):
        save_checkpoint(model=model, output_dir=checkpoint_tmp_path, metadata=metadata)
