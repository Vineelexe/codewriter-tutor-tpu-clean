from __future__ import annotations

import os

from cw360.teacher.providers.openai_compatible import OpenAICompatibleTeacherClient


class NvidiaNIMTeacherClient(OpenAICompatibleTeacherClient):
    def __init__(
        self,
        *,
        model_name: str,
        base_url: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        super().__init__(
            model_name=model_name,
            api_key_env="NVIDIA_API_KEY",
            base_url=base_url or os.environ.get("NVIDIA_BASE_URL") or "https://integrate.api.nvidia.com/v1",
            timeout_seconds=timeout_seconds,
            provider_name="nvidia_nim",
        )
