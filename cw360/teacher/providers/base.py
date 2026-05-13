from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TeacherResponse:
    text: str
    provider_name: str
    model_name: str
    request_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class BaseTeacherClient(ABC):
    provider_name: str
    model_name: str

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> TeacherResponse:
        raise NotImplementedError

    async def async_generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> TeacherResponse:
        return self.generate(prompt, system_prompt, temperature, max_tokens)

    @abstractmethod
    def validate_environment(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return None
