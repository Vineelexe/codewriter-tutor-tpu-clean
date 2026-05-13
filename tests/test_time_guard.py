from __future__ import annotations

from cw360.train.time_guard import PeriodicSaveTimer, SaveBeforeExitTimeGuard


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_time_guard_triggers_once_inside_save_margin() -> None:
    clock = FakeClock()
    guard = SaveBeforeExitTimeGuard(
        max_runtime_seconds=100.0,
        save_margin_seconds=10.0,
        clock=clock,
    )

    clock.now = 89.0
    assert not guard.should_save_before_exit()

    clock.now = 90.0
    assert guard.should_save_before_exit()
    assert not guard.should_save_before_exit()


def test_time_guard_reports_remaining_time() -> None:
    clock = FakeClock()
    guard = SaveBeforeExitTimeGuard(
        max_runtime_seconds=50.0,
        save_margin_seconds=5.0,
        clock=clock,
    )

    clock.now = 12.5

    assert guard.elapsed_seconds() == 12.5
    assert guard.remaining_seconds() == 37.5


def test_periodic_save_timer_triggers_after_interval() -> None:
    clock = FakeClock()
    timer = PeriodicSaveTimer(interval_seconds=30.0, clock=clock)

    clock.now = 29.0
    assert not timer.should_save()

    clock.now = 30.0
    assert timer.should_save()
    timer.mark_saved()

    clock.now = 59.0
    assert not timer.should_save()
    clock.now = 60.0
    assert timer.should_save()
