from __future__ import annotations

import pytest

from cw360.checkpoint.validate import CheckpointValidationError, validate_metadata

from tests.test_checkpoint_metadata import sample_cursor, sample_metadata


def test_tpu_checkpoint_requires_exact_cursor_metadata() -> None:
    metadata = sample_metadata(backend="xla_tpu", platform="kaggle-tpu")

    with pytest.raises(CheckpointValidationError, match="tpu_data_cursor"):
        validate_metadata(metadata)


def test_tpu_cursor_must_match_checkpoint_progress() -> None:
    metadata = sample_metadata(
        backend="xla_tpu",
        platform="kaggle-tpu",
        tpu_data_cursor=sample_cursor(global_step=6),
    )

    with pytest.raises(CheckpointValidationError, match="global_step"):
        validate_metadata(metadata)


def test_tpu_cursor_requires_drop_last_for_static_shapes() -> None:
    metadata = sample_metadata(
        backend="xla_tpu",
        platform="kaggle-tpu",
        tpu_data_cursor=sample_cursor(drop_last=False),
    )

    with pytest.raises(CheckpointValidationError, match="drop_last"):
        validate_metadata(metadata)
