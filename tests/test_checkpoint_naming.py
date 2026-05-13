from __future__ import annotations

from cw360.checkpoint import checkpoint_filename, parse_checkpoint_filename

from tests.test_checkpoint_metadata import sample_metadata


def test_checkpoint_filename_matches_project_convention() -> None:
    metadata = sample_metadata(
        model_size_label="420m",
        step=50_000,
        train_loss=4.231,
        trained_by="sarang",
        platform="kaggle-tpu-background",
    )

    filename = checkpoint_filename(metadata)

    assert (
        filename
        == "cw360_420m_step00050000_loss4.231_stage_base_pretrain_train_sarang_"
        "2026-05-23_kaggle-tpu-background.pt"
    )


def test_checkpoint_filename_parse_round_trip() -> None:
    filename = checkpoint_filename(sample_metadata(step=17, train_loss=3.999))

    parsed = parse_checkpoint_filename(filename)

    assert parsed.model_size_label == "tiny"
    assert parsed.step == 17
    assert parsed.train_loss == 3.999
    assert parsed.training_stage == "base_pretrain"
    assert parsed.trainer == "vineel"
    assert parsed.platform == "local-cpu-test"
