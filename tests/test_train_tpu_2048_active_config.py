from __future__ import annotations

from pathlib import Path

from cw360.config import load_config
from cw360.prepack.config import load_prepack_config
from cw360.train.shard_dataloader import validate_prepacked_manifest_for_training
from cw360.train.validate import validate_tpu_train_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_active_2048_tpu_train_config_is_not_later_gated() -> None:
    config = validate_tpu_train_config(CONFIG_DIR / "train_tpu_354m_2048.yaml")
    prepack = load_prepack_config(CONFIG_DIR / "prepack_tpu_2048.yaml")

    assert config["max_seq_len"] == 2048
    assert "later_stage_only" not in config
    assert "stage_gate" not in config
    assert config["dynamic_padding"] is False
    assert config["tokenizer_id"] == "bigcode/starcoder2-15b"
    assert prepack.prepack.seq_len_plus_one == 2049
    assert config["max_seq_len"] + 1 == prepack.prepack.seq_len_plus_one


def test_active_2048_prepacked_manifest_shape_contract_is_training_compatible() -> None:
    train_config = load_config(CONFIG_DIR / "train_tpu_354m_2048.yaml")
    prepack = load_prepack_config(CONFIG_DIR / "prepack_tpu_2048.yaml")
    manifest = {
        "max_seq_len": train_config["max_seq_len"],
        "seq_len_plus_one": prepack.prepack.seq_len_plus_one,
        "dtype": prepack.prepack.token_dtype,
        "shard_format": prepack.prepack.shard_format,
    }

    validate_prepacked_manifest_for_training(manifest)
