from __future__ import annotations

from dataclasses import dataclass

from cw360.constants import VALID_TRAINING_STAGES


@dataclass(frozen=True, slots=True)
class TrainingStageSpec:
    name: str
    trainable: bool
    requires_fim_format: bool
    instruction_weighted: bool


_STAGE_SPECS: dict[str, TrainingStageSpec] = {
    "base_pretrain": TrainingStageSpec(
        name="base_pretrain",
        trainable=True,
        requires_fim_format=False,
        instruction_weighted=False,
    ),
    "fim_train": TrainingStageSpec(
        name="fim_train",
        trainable=True,
        requires_fim_format=True,
        instruction_weighted=False,
    ),
    "instruction_tune": TrainingStageSpec(
        name="instruction_tune",
        trainable=True,
        requires_fim_format=False,
        instruction_weighted=True,
    ),
    "eval_only": TrainingStageSpec(
        name="eval_only",
        trainable=False,
        requires_fim_format=False,
        instruction_weighted=False,
    ),
}


def validate_training_stage(stage: str) -> str:
    if stage not in VALID_TRAINING_STAGES:
        raise ValueError(f"training_stage must be one of {sorted(VALID_TRAINING_STAGES)}")
    if stage not in _STAGE_SPECS:
        raise ValueError(f"training_stage has no train spec: {stage}")
    return stage


def get_training_stage_spec(stage: str) -> TrainingStageSpec:
    return _STAGE_SPECS[validate_training_stage(stage)]


def ensure_trainable_stage(stage: str) -> TrainingStageSpec:
    spec = get_training_stage_spec(stage)
    if not spec.trainable:
        raise ValueError("eval_only is not a trainable stage")
    return spec
