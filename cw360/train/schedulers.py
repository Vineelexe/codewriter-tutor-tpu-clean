from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch


ScheduleType = Literal["wsd", "cosine"]


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    total_steps: int
    warmup_steps: int = 0
    stable_steps: int | None = None
    min_lr_ratio: float = 0.1
    schedule_type: ScheduleType = "wsd"


class WarmupStableDecayScheduler:
    """Small stateful LR scheduler with explicit resume semantics."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: SchedulerConfig,
        *,
        last_step: int = 0,
    ) -> None:
        _validate_config(config)
        if last_step < 0:
            raise ValueError("last_step must be non-negative")
        self.optimizer = optimizer
        self.config = config
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.last_step = int(last_step)
        self._apply_lr_for_next_step()

    def lr_multiplier(self, step: int) -> float:
        if step < 0:
            raise ValueError("step must be non-negative")
        if step == 0:
            return 0.0 if self.config.warmup_steps > 0 else 1.0

        if self.config.warmup_steps > 0 and step <= self.config.warmup_steps:
            return step / self.config.warmup_steps

        stable_until = self._stable_until_step()
        if step <= stable_until:
            return 1.0

        decay_steps = max(1, self.config.total_steps - stable_until)
        progress = min(1.0, max(0.0, (step - stable_until) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.config.min_lr_ratio + (1.0 - self.config.min_lr_ratio) * cosine

    def step(self) -> None:
        self.last_step += 1
        self._apply_lr_for_next_step()

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "base_lrs": list(self.base_lrs),
            "last_step": self.last_step,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        config_data = state_dict.get("config")
        if isinstance(config_data, dict):
            self.config = SchedulerConfig(**config_data)
            _validate_config(self.config)
        self.base_lrs = [float(value) for value in state_dict["base_lrs"]]
        self.last_step = int(state_dict["last_step"])
        if len(self.base_lrs) != len(self.optimizer.param_groups):
            raise ValueError("scheduler param group count does not match optimizer")
        self._apply_lr_for_next_step()

    def _apply_lr_for_next_step(self) -> None:
        next_step = min(self.last_step + 1, self.config.total_steps)
        multiplier = self.lr_multiplier(next_step)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base_lr * multiplier

    def _stable_until_step(self) -> int:
        if self.config.stable_steps is not None:
            return self.config.warmup_steps + self.config.stable_steps
        if self.config.schedule_type == "cosine":
            return self.config.warmup_steps
        return max(self.config.warmup_steps, int(self.config.total_steps * 0.9))


def _validate_config(config: SchedulerConfig) -> None:
    if config.total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if config.stable_steps is not None and config.stable_steps < 0:
        raise ValueError("stable_steps must be non-negative")
    if not 0.0 <= config.min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1]")
    if config.schedule_type not in {"wsd", "cosine"}:
        raise ValueError("schedule_type must be 'wsd' or 'cosine'")
