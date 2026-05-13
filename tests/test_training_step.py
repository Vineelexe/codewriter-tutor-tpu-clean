from __future__ import annotations

import yaml

from cw360.checkpoint import read_metadata_json
from cw360.checkpoint.metadata import metadata_path_for_checkpoint
from cw360.train.cpu_trainer import CPUTinyTrainer


def test_cpu_tiny_training_step_saves_checkpoint(tmp_path) -> None:
    config = tiny_train_config(tmp_path, max_steps=1)
    trainer = CPUTinyTrainer(config=config, output_dir=tmp_path / "ckpts")

    before = [parameter.detach().clone() for parameter in trainer.model.parameters()]
    result = trainer.train()

    assert result.state.step == 1
    assert result.state.tokens_seen == 2 * 8
    assert result.state.sequences_seen == 2
    assert len(result.checkpoints) == 1
    assert any(
        not parameter.detach().equal(saved)
        for parameter, saved in zip(trainer.model.parameters(), before, strict=True)
    )

    metadata = read_metadata_json(metadata_path_for_checkpoint(result.checkpoints[-1]))
    assert metadata.step == 1
    assert metadata.tokens_seen == result.state.tokens_seen
    assert metadata.sequences_seen == result.state.sequences_seen


def tiny_train_config(tmp_path, *, max_steps: int = 2) -> dict[str, object]:
    model_config_path = tmp_path / "model_tiny_unit.yaml"
    model_config_path.write_text(
        yaml.safe_dump(
            {
                "config_type": "model",
                "model_name": "cw360",
                "size_label": "tiny",
                "tokenizer_name": "bigcode/starcoder2-15b",
                "vocab_size": 32,
                "max_position_embeddings": 16,
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "intermediate_size": 32,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return {
        "config_type": "train",
        "training_stage": "base_pretrain",
        "platform": "local-cpu-test",
        "model_config_path": str(model_config_path),
        "max_seq_len": 8,
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "max_steps": max_steps,
        "local_cpu_train_seq_len": 8,
        "drop_last": True,
        "static_shapes": True,
    }
