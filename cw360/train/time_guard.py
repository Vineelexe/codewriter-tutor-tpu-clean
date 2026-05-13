from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(slots=True)
class SaveBeforeExitTimeGuard:
    max_runtime_seconds: float
    save_margin_seconds: float
    clock: Callable[[], float] = time.monotonic
    started_at: float = field(init=False)
    _triggered: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.max_runtime_seconds <= 0:
            raise ValueError("max_runtime_seconds must be positive")
        if self.save_margin_seconds < 0:
            raise ValueError("save_margin_seconds must be non-negative")
        self.started_at = float(self.clock())

    def elapsed_seconds(self) -> float:
        return max(0.0, float(self.clock()) - self.started_at)

    def remaining_seconds(self) -> float:
        return self.max_runtime_seconds - self.elapsed_seconds()

    def should_save_before_exit(self) -> bool:
        if self._triggered:
            return False
        if self.remaining_seconds() <= self.save_margin_seconds:
            self._triggered = True
            return True
        return False

    def mark_saved(self) -> None:
        self._triggered = True


@dataclass(slots=True)
class PeriodicSaveTimer:
    interval_seconds: float
    clock: Callable[[], float] = time.monotonic
    last_save_at: float = field(init=False)

    def __post_init__(self) -> None:
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.last_save_at = float(self.clock())

    def should_save(self) -> bool:
        return float(self.clock()) - self.last_save_at >= self.interval_seconds

    def mark_saved(self) -> None:
        self.last_save_at = float(self.clock())
