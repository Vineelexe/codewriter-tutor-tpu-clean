from __future__ import annotations

import pytest

from cw360.checkpoint import find_latest_checkpoint, save_checkpoint
from cw360.model import CodeWriterTutorLM

from tests.test_checkpoint_metadata import sample_metadata, tiny_lm_config


def test_find_latest_checkpoint_returns_highest_valid_step(checkpoint_tmp_path) -> None:
    model = CodeWriterTutorLM(tiny_lm_config(), init_seed=1)
    first = save_checkpoint(model=model, output_dir=checkpoint_tmp_path, metadata=sample_metadata(step=1))
    second = save_checkpoint(
        model=model,
        output_dir=checkpoint_tmp_path,
        metadata=sample_metadata(step=9, train_loss=3.75),
    )
    assert first is not None
    assert second is not None
    (checkpoint_tmp_path / ".cw360_tiny_step99999999_loss1.000_stage_base_pretrain.tmp").write_text(
        "incomplete",
        encoding="utf-8",
    )

    latest = find_latest_checkpoint(checkpoint_tmp_path)

    assert latest is not None
    assert latest.path == second
    assert latest.metadata.step == 9


def test_find_latest_checkpoint_warns_and_ignores_missing_metadata(checkpoint_tmp_path) -> None:
    missing_metadata = checkpoint_tmp_path / (
        "cw360_tiny_step00000011_loss4.231_stage_base_pretrain_train_vineel_"
        "2026-05-23_local-cpu-test.pt"
    )
    missing_metadata.write_bytes(b"not a torch checkpoint")

    with pytest.warns(UserWarning, match="without matching metadata"):
        latest = find_latest_checkpoint(checkpoint_tmp_path)

    assert latest is None
