from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock


@dataclass(slots=True)
class RateLimitConfig:
    requests_per_minute: int | None = None


class RateLimiter:
    def __init__(
        self,
        config: RateLimitConfig | None = None,
        *,
        sleep_fn: Callable[[float], None] = time.sleep,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or RateLimitConfig()
        self.sleep_fn = sleep_fn
        self.monotonic_fn = monotonic_fn
        self._lock = Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        rpm = self.config.requests_per_minute
        if not rpm or rpm <= 0:
            return
        interval = 60.0 / rpm
        with self._lock:
            now = self.monotonic_fn()
            if now < self._next_allowed:
                self.sleep_fn(self._next_allowed - now)
                now = self.monotonic_fn()
            self._next_allowed = max(now, self._next_allowed) + interval
