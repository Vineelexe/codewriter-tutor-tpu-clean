from __future__ import annotations

import torch

from cw360.checkpoint import load_checkpoint, metadata_path_for_checkpoint, save_checkpoint
from cw360.model import CodeWriterTutorLM

from tests.test_checkpoint_metadata import sample_metadata, tiny_lm_config


def test_checkpoint_save_load_restores_model_optimizer_scheduler(checkpoint_tmp_path) -> None:
    config = tiny_lm_config()
    model = CodeWriterTutorLM(config, init_seed=7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    loss = model(input_ids, labels=input_ids).loss
    assert loss is not None
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    metadata = sample_metadata(train_loss=float(loss.item()))
    checkpoint = save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=checkpoint_tmp_path,
        metadata=metadata,
    )
    assert checkpoint is not None
    assert checkpoint.exists()
    assert metadata_path_for_checkpoint(checkpoint).exists()

    restored = CodeWriterTutorLM(config, init_seed=99)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1.0e-3)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
        restored_optimizer,
        lr_lambda=lambda _: 1.0,
    )

    loaded_metadata, cursor = load_checkpoint(
        checkpoint_path=checkpoint,
        model=restored,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        expected_training_stage="base_pretrain",
    )

    assert loaded_metadata == metadata
    assert cursor is None
    for saved, loaded in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(saved, loaded)
    assert restored_scheduler.last_epoch == scheduler.last_epoch


def test_checkpoint_load_rejects_architecture_mismatch(checkpoint_tmp_path) -> None:
    config = tiny_lm_config()
    model = CodeWriterTutorLM(config, init_seed=7)
    checkpoint = save_checkpoint(model=model, output_dir=checkpoint_tmp_path, metadata=sample_metadata())
    assert checkpoint is not None

    wrong_model = CodeWriterTutorLM(tiny_lm_config(hidden_size=24), init_seed=7)

    try:
        load_checkpoint(checkpoint_path=checkpoint, model=wrong_model)
    except ValueError as exc:
        assert "architecture mismatch" in str(exc)
    else:
        raise AssertionError("architecture mismatch was not rejected")
