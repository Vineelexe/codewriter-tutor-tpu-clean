from __future__ import annotations

from cw360.data.schemas import TrainingExample
from cw360.prepack.writer import sample_examples_by_loss_weight


def test_loss_weight_sampling_oversamples_before_writing() -> None:
    examples = [
        TrainingExample(
            text="def f(): pass",
            source="synthetic",
            example_type="instruction",
            loss_weight=2.0,
        )
    ]

    sampled = list(sample_examples_by_loss_weight(examples, tier_weights={}, seed=1))

    assert len(sampled) == 2
    assert all(item.metadata["prepack_sampling_multiplier"] == 2.0 for item in sampled)


def test_zero_loss_weight_skips_example_before_runtime_loss() -> None:
    examples = [
        TrainingExample(
            text="def f(): pass",
            source="synthetic",
            example_type="instruction",
            loss_weight=0.0,
        )
    ]

    assert list(sample_examples_by_loss_weight(examples, tier_weights={}, seed=1)) == []
