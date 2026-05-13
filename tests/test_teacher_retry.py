from __future__ import annotations

import pytest

from cw360.teacher.retry import RetryConfig, RetryExhaustedError, run_with_retries


def test_retry_succeeds_after_retryable_failure() -> None:
    calls = {"count": 0}

    def flaky() -> str:
        calls["count"] += 1
        if calls["count"] < 2:
            raise TimeoutError("temporary")
        return "ok"

    result = run_with_retries(
        flaky,
        config=RetryConfig(max_retries=2, initial_delay_seconds=0),
        sleep_fn=lambda _: None,
    )

    assert result == "ok"
    assert calls["count"] == 2


def test_retry_raises_after_exhaustion() -> None:
    with pytest.raises(RetryExhaustedError):
        run_with_retries(
            lambda: (_ for _ in ()).throw(TimeoutError("still down")),
            config=RetryConfig(max_retries=1, initial_delay_seconds=0),
            sleep_fn=lambda _: None,
        )
