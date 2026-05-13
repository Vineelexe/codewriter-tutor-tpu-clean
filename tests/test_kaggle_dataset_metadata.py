from __future__ import annotations

from cw360.prepack.kaggle_dataset import (
    build_kaggle_dataset_metadata,
    write_kaggle_dataset_metadata,
)
from tests.test_prepack_writer import small_prepack_config


def test_kaggle_dataset_metadata_uses_dataset_name_and_private_owner(tmp_path) -> None:
    config = small_prepack_config(tmp_path)

    metadata = build_kaggle_dataset_metadata(config)

    assert metadata["title"] == "cw-tiny-prepack"
    assert metadata["id"] == "vineel/cw-tiny-prepack"
    assert metadata["licenses"] == [{"name": "other"}]


def test_kaggle_dataset_metadata_is_written_to_output_dir(tmp_path) -> None:
    config = small_prepack_config(tmp_path)

    metadata = write_kaggle_dataset_metadata(tmp_path, config)

    assert (tmp_path / "dataset-metadata.json").exists()
    assert metadata["id"].endswith("/cw-tiny-prepack")
