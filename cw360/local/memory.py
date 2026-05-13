from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import psutil

from cw360.config import ModelConfig
from cw360.model.count import build_parameter_report

SPEED_BOTTLENECK_WARNING = (
    "CPU inference may fit in memory while still being slow; speed is likely the main "
    "laptop bottleneck for full candidate models."
)


@dataclass(frozen=True, slots=True)
class MemoryEstimate:
    precision: str
    bytes_per_parameter: float
    parameter_bytes: int
    theoretical: bool = False

    @property
    def gib(self) -> float:
        return self.parameter_bytes / (1024**3)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gib"] = self.gib
        return payload


@dataclass(frozen=True, slots=True)
class LocalMemoryReport:
    size_label: str
    total_parameters: int
    available_ram_bytes: int
    estimates: tuple[MemoryEstimate, ...]
    warning: str

    @property
    def available_ram_gib(self) -> float:
        return self.available_ram_bytes / (1024**3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "size_label": self.size_label,
            "total_parameters": self.total_parameters,
            "available_ram_bytes": self.available_ram_bytes,
            "available_ram_gib": self.available_ram_gib,
            "estimates": [estimate.to_dict() for estimate in self.estimates],
            "warning": self.warning,
        }


def build_memory_estimates(total_parameters: int) -> tuple[MemoryEstimate, ...]:
    if total_parameters <= 0:
        raise ValueError("total_parameters must be positive")
    return (
        _estimate("fp32", total_parameters, 4.0),
        _estimate("bf16_fp16", total_parameters, 2.0),
        _estimate("int8", total_parameters, 1.0),
        _estimate("4bit_theoretical", total_parameters, 0.5, theoretical=True),
    )


def build_local_memory_report(
    config: ModelConfig,
    *,
    available_ram_bytes: int | None = None,
) -> LocalMemoryReport:
    parameter_report = build_parameter_report(config)
    available = (
        int(available_ram_bytes)
        if available_ram_bytes is not None
        else int(psutil.virtual_memory().available)
    )
    return LocalMemoryReport(
        size_label=config.size_label,
        total_parameters=parameter_report.total_parameters,
        available_ram_bytes=available,
        estimates=build_memory_estimates(parameter_report.total_parameters),
        warning=SPEED_BOTTLENECK_WARNING,
    )


def _estimate(
    precision: str,
    total_parameters: int,
    bytes_per_parameter: float,
    *,
    theoretical: bool = False,
) -> MemoryEstimate:
    return MemoryEstimate(
        precision=precision,
        bytes_per_parameter=bytes_per_parameter,
        parameter_bytes=int(total_parameters * bytes_per_parameter),
        theoretical=theoretical,
    )
