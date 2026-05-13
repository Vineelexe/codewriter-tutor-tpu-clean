from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class RetryConfig:
    max_retries: int = 5
    initial_delay_seconds: float = 0.5
    max_delay_seconds: float = 20.0
    backoff_factor: float = 2.0


class RetryExhaustedError(RuntimeError):
    pass


def run_with_retries(
    operation: Callable[[], T],
    *,
    config: RetryConfig,
    retryable: tuple[type[BaseException], ...] = (Exception,),
    sleep_fn: Callable[[float], None] = time.sleep,
) -> T:
    delay = config.initial_delay_seconds
    attempts = max(1, config.max_retries + 1)
    last_error: BaseException | None = None
    for attempt in range(attempts):
        try:
            return operation()
        except retryable as exc:
            last_error = exc
            if attempt == attempts - 1:
                break
            sleep_fn(min(delay, config.max_delay_seconds))
            delay *= config.backoff_factor
    message = f"operation failed after {attempts} attempts: {last_error}"
    raise RetryExhaustedError(message) from last_error
