from __future__ import annotations

import os
from typing import Any

import httpx

from cw360.teacher.providers.base import BaseTeacherClient, TeacherResponse


class GeminiTeacherClient(BaseTeacherClient):
    provider_name = "gemini"

    def __init__(self, *, model_name: str, timeout_seconds: float = 60.0) -> None:
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self._client = httpx.Client(timeout=timeout_seconds)

    def validate_environment(self) -> None:
        if not os.environ.get("GEMINI_API_KEY"):
            raise RuntimeError("GEMINI_API_KEY is required for provider gemini")

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> TeacherResponse:
        self.validate_environment()
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:generateContent"
        )
        response = self._client.post(
            url,
            params={"key": os.environ["GEMINI_API_KEY"]},
            json={
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                },
            },
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        text = str(data["candidates"][0]["content"]["parts"][0]["text"])
        return TeacherResponse(
            text=text,
            provider_name=self.provider_name,
            model_name=self.model_name,
            request_id=str(data.get("responseId") or ""),
            raw={"usage": data.get("usageMetadata")},
        )

    def close(self) -> None:
        self._client.close()
