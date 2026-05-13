from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StepMetrics:
    loss: float
    learning_rate: float
    step: int
    tokens_seen: int
    sequences_seen: int
    tokens_per_second: float
    sequences_per_second: float
    step_time: float


class MetricsTracker:
    def __init__(self, *, clock: callable | None = None) -> None:
        self._clock = clock or time.perf_counter
        self._last_timestamp = float(self._clock())
        self.history: list[StepMetrics] = []

    def record_step(
        self,
        *,
        loss: float,
        learning_rate: float,
        step: int,
        tokens_seen: int,
        sequences_seen: int,
        step_tokens: int,
        step_sequences: int,
        step_time: float | None = None,
    ) -> StepMetrics:
        if step_time is None:
            now = float(self._clock())
            step_time = max(now - self._last_timestamp, 1.0e-12)
            self._last_timestamp = now
        else:
            step_time = max(float(step_time), 1.0e-12)

        metrics = StepMetrics(
            loss=float(loss),
            learning_rate=float(learning_rate),
            step=int(step),
            tokens_seen=int(tokens_seen),
            sequences_seen=int(sequences_seen),
            tokens_per_second=int(step_tokens) / step_time,
            sequences_per_second=int(step_sequences) / step_time,
            step_time=step_time,
        )
        self.history.append(metrics)
        return metrics
