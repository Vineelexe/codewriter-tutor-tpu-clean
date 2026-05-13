from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from cw360.constants import VALID_TRAINING_STAGES

BASE_TARGET_WEIGHTS: dict[str, float] = {
    "stack_v2_python": 0.30,
    "codesearchnet_python": 0.25,
    "debugbench": 0.20,
    "synthetic": 0.15,
    "fineweb_edu": 0.05,
    "cosmopedia_cs": 0.05,
}

CHEAP_BASE_SOURCES = frozenset(
    {
        "stack_v2_python",
        "codesearchnet_python",
        "fineweb_edu",
        "cosmopedia_cs",
    }
)
DEBUG_INSTRUCTION_SOURCES = frozenset({"debugbench", "synthetic"})


@dataclass(frozen=True, slots=True)
class StageMixture:
    training_stage: str
    weights: MappingProxyType[str, float]
    description: str
    requires_precomputed_parent_data: bool = False

    @property
    def normalized_weights(self) -> dict[str, float]:
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError(f"stage mixture {self.training_stage} has no positive weight")
        return {name: weight / total for name, weight in self.weights.items()}

    @property
    def cheap_base_ratio(self) -> float:
        normalized = self.normalized_weights
        return sum(normalized.get(source, 0.0) for source in CHEAP_BASE_SOURCES)

    @property
    def debug_instruction_ratio(self) -> float:
        normalized = self.normalized_weights
        return sum(normalized.get(source, 0.0) for source in DEBUG_INSTRUCTION_SOURCES)


def _frozen(weights: dict[str, float]) -> MappingProxyType[str, float]:
    return MappingProxyType(dict(weights))


STAGE_MIXTURES: dict[str, StageMixture] = {
    "base_pretrain": StageMixture(
        training_stage="base_pretrain",
        weights=_frozen(BASE_TARGET_WEIGHTS),
        description=(
            "Base target curriculum: 65% cheap locally scrubbed code/education data and "
            "35% debugging/instruction data already prepared outside TPU training."
        ),
        requires_precomputed_parent_data=True,
    ),
    "fim_train": StageMixture(
        training_stage="fim_train",
        weights=_frozen(
            {
                "stack_v2_python": 0.45,
                "codesearchnet_python": 0.35,
                "debugbench": 0.10,
                "synthetic": 0.05,
                "fineweb_edu": 0.025,
                "cosmopedia_cs": 0.025,
            }
        ),
        description="FIM-heavy pass biased toward Python source and docstring/code pairs.",
        requires_precomputed_parent_data=True,
    ),
    "instruction_tune": StageMixture(
        training_stage="instruction_tune",
        weights=_frozen(
            {
                "debugbench": 0.45,
                "synthetic": 0.45,
                "codesearchnet_python": 0.05,
                "stack_v2_python": 0.05,
            }
        ),
        description="Instruction pass over precomputed debugging and code-writing examples.",
        requires_precomputed_parent_data=True,
    ),
    "eval_only": StageMixture(
        training_stage="eval_only",
        weights=_frozen(
            {
                "debugbench": 0.40,
                "synthetic": 0.35,
                "codesearchnet_python": 0.15,
                "stack_v2_python": 0.10,
            }
        ),
        description="Evaluation stream over held-out Python writing and debugging examples.",
        requires_precomputed_parent_data=False,
    ),
}


def get_stage_mixture(training_stage: str) -> StageMixture:
    if training_stage not in VALID_TRAINING_STAGES:
        raise ValueError(
            f"training_stage must be one of {sorted(VALID_TRAINING_STAGES)}, got {training_stage!r}"
        )
    try:
        return STAGE_MIXTURES[training_stage]
    except KeyError as exc:
        raise ValueError(f"no mixture configured for training_stage {training_stage!r}") from exc


def stage_weights(training_stage: str) -> dict[str, float]:
    return dict(get_stage_mixture(training_stage).weights)
