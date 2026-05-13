from __future__ import annotations

from cw360.teacher.providers.base import BaseTeacherClient, TeacherResponse
from cw360.teacher.providers.gemini import GeminiTeacherClient
from cw360.teacher.providers.mock import MockTeacherClient
from cw360.teacher.providers.nvidia_nim import NvidiaNIMTeacherClient
from cw360.teacher.providers.openai_compatible import OpenAICompatibleTeacherClient

__all__ = [
    "BaseTeacherClient",
    "GeminiTeacherClient",
    "MockTeacherClient",
    "NvidiaNIMTeacherClient",
    "OpenAICompatibleTeacherClient",
    "TeacherResponse",
]
