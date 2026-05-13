from __future__ import annotations

import torch

from cw360.train.schedulers import SchedulerConfig, WarmupStableDecayScheduler


def test_wsd_scheduler_warmup_stable_decay_and_resume() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = WarmupStableDecayScheduler(
        optimizer,
        SchedulerConfig(total_steps=10, warmup_steps=2, stable_steps=3, min_lr_ratio=0.1),
    )

    assert optimizer.param_groups[0]["lr"] == 0.5
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 1.0
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 1.0

    for _ in range(3):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] < 1.0

    state = scheduler.state_dict()
    restored_optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.tensor([2.0]))], lr=1.0)
    restored = WarmupStableDecayScheduler(
        restored_optimizer,
        SchedulerConfig(total_steps=1),
    )
    restored.load_state_dict(state)

    assert restored.last_step == scheduler.last_step
    assert restored.get_last_lr() == scheduler.get_last_lr()


def test_cosine_scheduler_decays_after_warmup() -> None:
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.tensor([1.0]))], lr=2.0)
    scheduler = WarmupStableDecayScheduler(
        optimizer,
        SchedulerConfig(
            total_steps=4,
            warmup_steps=1,
            min_lr_ratio=0.25,
            schedule_type="cosine",
        ),
    )

    assert scheduler.get_last_lr() == [2.0]
    scheduler.step()
    scheduler.step()

    assert 0.5 < scheduler.get_last_lr()[0] < 2.0
