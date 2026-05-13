from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TrainingState:
    step: int = 0
    tokens_seen: int = 0
    sequences_seen: int = 0

    def advance(self, *, tokens: int, sequences: int, steps: int = 1) -> None:
        if steps <= 0:
            raise ValueError("steps must be positive")
        if tokens < 0 or sequences < 0:
            raise ValueError("tokens and sequences must be non-negative")
        self.step += steps
        self.tokens_seen += tokens
        self.sequences_seen += sequences

    @classmethod
    def from_checkpoint_values(
        cls,
        *,
        step: int,
        tokens_seen: int,
        sequences_seen: int,
    ) -> "TrainingState":
        if step < 0 or tokens_seen < 0 or sequences_seen < 0:
            raise ValueError("checkpoint progress values must be non-negative")
        return cls(
            step=int(step),
            tokens_seen=int(tokens_seen),
            sequences_seen=int(sequences_seen),
        )
