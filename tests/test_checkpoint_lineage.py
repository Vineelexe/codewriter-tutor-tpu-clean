from __future__ import annotations

import json

from cw360.checkpoint import build_handoff_manifest, save_checkpoint, write_handoff_manifest
from cw360.model import CodeWriterTutorLM
from cw360.utils.hashing import sha256_file

from tests.test_checkpoint_metadata import sample_cursor, sample_metadata, tiny_lm_config


def test_handoff_manifest_records_checkpoint_hash_and_resume_identity(checkpoint_tmp_path) -> None:
    model = CodeWriterTutorLM(tiny_lm_config(), init_seed=1)
    metadata = sample_metadata(
        platform="kaggle-tpu-background",
        backend="xla_tpu",
        tpu_data_cursor=sample_cursor(),
    )
    checkpoint = save_checkpoint(model=model, output_dir=checkpoint_tmp_path, metadata=metadata)
    assert checkpoint is not None

    manifest = build_handoff_manifest(
        checkpoint_path=checkpoint,
        trainer_leaving="vineel",
        next_trainer_expected="sarang",
        exact_resume_command="python train_tpu.py --resume path/to/checkpoint.pt",
    )

    assert manifest.latest_checkpoint_path == str(checkpoint)
    assert manifest.checkpoint_sha256 == sha256_file(checkpoint)
    assert manifest.trainer_leaving == "vineel"
    assert manifest.next_trainer_expected == "sarang"
    assert manifest.prepacked_dataset_identity["prepacked_dataset_name"] == "cw360-prepacked"
    assert manifest.tpu_data_cursor is not None
    assert manifest.tpu_data_cursor["shard_index"] == 2


def test_write_handoff_manifest_never_overwrites(checkpoint_tmp_path) -> None:
    model = CodeWriterTutorLM(tiny_lm_config(), init_seed=1)
    checkpoint = save_checkpoint(model=model, output_dir=checkpoint_tmp_path, metadata=sample_metadata(step=8))
    assert checkpoint is not None
    output_path = checkpoint_tmp_path / "handoff.json"

    written = write_handoff_manifest(
        checkpoint_path=checkpoint,
        trainer_leaving="vineel",
        next_trainer_expected="sarang",
        exact_resume_command="python train.py --resume checkpoint.pt",
        output_path=output_path,
    )

    assert written == output_path
    raw = json.loads(output_path.read_text(encoding="utf-8"))
    assert raw["next_trainer_expected"] == "sarang"
    try:
        write_handoff_manifest(
            checkpoint_path=checkpoint,
            trainer_leaving="vineel",
            next_trainer_expected="sarang",
            exact_resume_command="python train.py --resume checkpoint.pt",
            output_path=output_path,
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("handoff manifest overwrite was not rejected")
