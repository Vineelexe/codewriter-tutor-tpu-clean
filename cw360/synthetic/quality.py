from __future__ import annotations

import re
from difflib import SequenceMatcher

from cw360.synthetic.schemas import StructuredSyntheticRecord

MIN_PROMPT_CHARS = 12
MIN_RESPONSE_CHARS = 30
MAX_RESPONSE_CHARS = 30_000

PYTHON_MARKERS = (
    "python",
    "def ",
    "class ",
    "pytest",
    "traceback",
    "exception",
    "list comprehension",
    "dictionary",
    "pandas",
    "numpy",
    "asyncio",
    ".py",
)
GENERIC_FLUFF = (
    "as an ai language model",
    "i hope this helps",
    "great question",
    "it depends on the context",
    "here is the answer",
)
PROVIDER_ERROR_MARKERS = (
    "api error",
    "rate limit",
    "429",
    "quota exceeded",
    "internal server error",
    "safety blocked",
    "model overloaded",
    "traceback (most recent call last)",
)
UNSAFE_MARKERS = (
    "steal api key",
    "exfiltrate",
    "malware",
    "ransomware",
    "delete system32",
    "credential theft",
    "bypass authentication",
)


def quality_rejection_reasons(record: StructuredSyntheticRecord) -> list[str]:
    reasons: list[str] = []
    prompt = record.user_prompt.strip()
    response = record.assistant_response.strip()
    combined = "\n".join(
        item
        for item in (
            prompt,
            response,
            record.code_before or "",
            record.code_after or "",
            record.explanation or "",
            record.tests or "",
            " ".join(record.skills),
        )
        if item
    )
    combined_norm = _normalize(combined)
    response_norm = _normalize(response)
    prompt_norm = _normalize(prompt)

    if len(prompt) < MIN_PROMPT_CHARS:
        reasons.append("prompt_too_short")
    if len(response) < MIN_RESPONSE_CHARS:
        reasons.append("response_too_short")
    if len(response) > MAX_RESPONSE_CHARS:
        reasons.append("response_too_long")
    if not _is_python_related(combined_norm, record):
        reasons.append("not_python_related")
    if _near_identical(prompt_norm, response_norm):
        reasons.append("response_nearly_identical_to_prompt")
    if _has_generic_fluff(response_norm):
        reasons.append("generic_fluff")
    if _has_provider_error(combined_norm):
        reasons.append("provider_error_text")
    if _has_unsafe_content(combined_norm):
        reasons.append("unsafe_or_malicious_content")
    if not _has_training_signal(response, record):
        reasons.append("no_useful_training_signal")
    return reasons


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def _is_python_related(text: str, record: StructuredSyntheticRecord) -> bool:
    if any(marker in text for marker in PYTHON_MARKERS):
        return True
    if record.source.lower() in {"stack_v2_python", "codesearchnet_python", "debugbench"}:
        return True
    return any(skill.lower() in {"python", "pytest", "debugging"} for skill in record.skills)


def _near_identical(prompt: str, response: str) -> bool:
    if not prompt or not response:
        return False
    return prompt == response or SequenceMatcher(None, prompt, response).ratio() >= 0.92


def _has_generic_fluff(response: str) -> bool:
    if any(marker in response for marker in GENERIC_FLUFF):
        return True
    return response in {"ok", "sure", "done", "yes"}


def _has_provider_error(text: str) -> bool:
    return any(marker in text for marker in PROVIDER_ERROR_MARKERS)


def _has_unsafe_content(text: str) -> bool:
    return any(marker in text for marker in UNSAFE_MARKERS)


def _has_training_signal(response: str, record: StructuredSyntheticRecord) -> bool:
    if record.code_after or record.tests or record.explanation:
        return True
    signal_markers = ("def ", "class ", "return ", "pytest", "assert ", "because", "fix", "bug")
    return any(marker in response.lower() for marker in signal_markers)
