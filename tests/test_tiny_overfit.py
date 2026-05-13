from __future__ import annotations

import torch

from cw360.train.cpu_trainer import CPUTinyTrainer
from cw360.train.trainer import TrainingBatch, repeat_batch

from tests.test_training_step import tiny_train_config


def test_tiny_model_overfits_one_fixed_batch_quickly(tmp_path) -> None:
    config = tiny_train_config(tmp_path, max_steps=100)
    config["learning_rate"] = 0.03
    config["max_grad_norm"] = 5.0
    tokens = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [9, 8, 7, 6, 5, 4, 3, 2, 1],
        ],
        dtype=torch.long,
    )
    batch = TrainingBatch(input_ids=tokens[:, :-1], labels=tokens[:, 1:])
    trainer = CPUTinyTrainer(
        config=config,
        output_dir=tmp_path / "ckpts",
        batch_factory=lambda _cursor: repeat_batch(batch),
    )

    result = trainer.train()

    # This tiny architecture reliably memorizes the fixed next-token batch, but
    # asserting zero-ish loss would make the CPU test unnecessarily brittle.
    assert result.final_loss < 0.5
    assert result.final_loss < result.metrics[0].loss * 0.2
