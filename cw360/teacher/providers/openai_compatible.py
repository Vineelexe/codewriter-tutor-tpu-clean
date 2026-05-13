from __future__ import annotations

import os
from typing import Any

import httpx

from cw360.teacher.providers.base import BaseTeacherClient, TeacherResponse


class OpenAICompatibleTeacherClient(BaseTeacherClient):
    provider_name = "openai_compatible"

    def __init__(
        self,
        *,
        model_name: str,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        timeout_seconds: float = 60.0,
        provider_name: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.api_key_env = api_key_env
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.timeout_seconds = timeout_seconds
        if provider_name:
            self.provider_name = provider_name
        self._client = httpx.Client(timeout=timeout_seconds)

    def validate_environment(self) -> None:
        if not os.environ.get(self.api_key_env):
            raise RuntimeError(f"{self.api_key_env} is required for provider {self.provider_name}")

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> TeacherResponse:
        self.validate_environment()
        response = self._client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {os.environ[self.api_key_env]}"},
            json={
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        text = str(data["choices"][0]["message"]["content"])
        return TeacherResponse(
            text=text,
            provider_name=self.provider_name,
            model_name=self.model_name,
            request_id=str(data.get("id") or ""),
            raw={"usage": data.get("usage")},
        )

    def close(self) -> None:
        self._client.close()
