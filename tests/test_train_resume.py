from __future__ import annotations

from cw360.checkpoint import read_metadata_json
from cw360.checkpoint.metadata import metadata_path_for_checkpoint
from cw360.train.cpu_trainer import CPUTinyTrainer

from tests.test_training_step import tiny_train_config


def test_cpu_tiny_training_resumes_model_optimizer_scheduler_and_counts(tmp_path) -> None:
    config = tiny_train_config(tmp_path, max_steps=1)
    first = CPUTinyTrainer(config=config, output_dir=tmp_path / "ckpts")
    first_result = first.train()

    resumed_config = tiny_train_config(tmp_path, max_steps=2)
    resumed = CPUTinyTrainer(
        config=resumed_config,
        output_dir=tmp_path / "ckpts",
        resume_checkpoint=first_result.checkpoints[-1],
    )
    second_result = resumed.train()

    assert second_result.resumed_from == first_result.checkpoints[-1]
    assert second_result.state.step == 2
    assert second_result.state.tokens_seen == 2 * first_result.state.tokens_seen
    assert second_result.state.sequences_seen == 2 * first_result.state.sequences_seen

    metadata = read_metadata_json(metadata_path_for_checkpoint(second_result.checkpoints[-1]))
    assert metadata.previous_checkpoint == str(first_result.checkpoints[-1])
    assert metadata.step == 2
