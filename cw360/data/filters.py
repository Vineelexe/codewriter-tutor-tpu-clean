from __future__ import annotations

import math
import re
from collections import Counter

from cw360.data.dedupe import ExactDeduper
from cw360.data.quality_tiers import tier_from_signal
from cw360.data.schemas import FilterDecision, TrainingExample

LOCKFILE_NAMES = {
    "package-lock.json",
    "yarn.lock",
    "poetry.lock",
    "pdm.lock",
    "requirements.txt",
    "Cargo.lock",
    "Gemfile.lock",
}
LICENSE_WORDS = {"copyright", "license", "permission", "warranty", "mit", "apache"}
PYTHON_SIGNALS = ("def ", "class ", "import ", "from ", "raise ", "return ", "pytest", "Traceback")
MALWARE_PATTERNS = (
    "os.system('rm -rf /",
    'os.system("rm -rf /',
    "subprocess.call(['rm', '-rf', '/'",
    "shutil.rmtree('/')",
    "socket.connect((",
    "keylogger",
    "credential_stealer",
)


def estimate_token_count(text: str) -> int:
    # Good enough for preview without importing a tokenizer or downloading assets.
    return max(1, math.ceil(len(text) / 4)) if text else 0


def looks_binary_or_corrupt(text: str) -> bool:
    if "\x00" in text:
        return True
    if not text:
        return False
    control_count = sum(1 for char in text if ord(char) < 32 and char not in "\n\r\t")
    replacement_count = text.count("\ufffd")
    return (control_count + replacement_count) / max(1, len(text)) > 0.02


def has_repeated_character_spam(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 80:
        return False
    most_common = Counter(compact).most_common(1)[0][1]
    return most_common / len(compact) > 0.65


def has_low_entropy(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 160:
        return False
    counts = Counter(compact)
    entropy = -sum(
        (count / len(compact)) * math.log2(count / len(compact)) for count in counts.values()
    )
    return entropy < 2.2


def has_repetition_spam(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 8:
        return False
    repeated = sum(count for count in Counter(lines).values() if count > 1)
    return repeated / len(lines) > 0.55


def has_very_long_unbroken_line(text: str, *, max_line_len: int = 5000) -> bool:
    for line in text.splitlines() or [text]:
        if len(line) > max_line_len and not re.search(r"\s", line):
            return True
    return False


def looks_minified(text: str) -> bool:
    lines = text.splitlines()
    if len(text) < 1000 or len(lines) > 4:
        return False
    punctuation = sum(1 for char in text if char in "{}[]();,:")
    return punctuation / max(1, len(text)) > 0.18


def looks_like_lockfile_or_dependency_dump(example: TrainingExample) -> bool:
    path = str(example.metadata.get("path") or example.metadata.get("file_name") or "")
    if path and path.replace("\\", "/").split("/")[-1] in LOCKFILE_NAMES:
        return True
    text = example.text
    if "version =" in text and "[[package]]" in text and "dependencies =" in text:
        return True
    requirement_lines = sum(1 for line in text.splitlines() if re.search(r"==\d|\>=\d|~=\d", line))
    return requirement_lines >= 20


def mostly_license_boilerplate(text: str) -> bool:
    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    license_lines = sum(any(word in line for word in LICENSE_WORDS) for line in lines)
    code_lines = sum(any(signal.strip() in line for signal in PYTHON_SIGNALS) for line in lines)
    return license_lines / len(lines) > 0.45 and code_lines < 3


def contains_dangerous_code(text: str) -> bool:
    lowered = text.lower()
    return any(pattern.lower() in lowered for pattern in MALWARE_PATTERNS)


def has_programming_signal(text: str) -> bool:
    lowered = text.lower()
    return any(signal.lower() in lowered for signal in PYTHON_SIGNALS) or any(
        word in lowered
        for word in (
            "python",
            "algorithm",
            "function",
            "debug",
            "stack trace",
            "complexity",
            "data structure",
        )
    )


def obvious_trash_filter(
    example: TrainingExample,
    *,
    deduper: ExactDeduper | None = None,
) -> FilterDecision | None:
    text = example.text
    if not text or not text.strip():
        return tier_from_signal(trash=True, reason="empty_text")
    if looks_binary_or_corrupt(text):
        return tier_from_signal(trash=True, reason="binary_or_corrupt_text")
    if deduper is not None and deduper.is_duplicate(text):
        return tier_from_signal(trash=True, reason="duplicate_content")
    if looks_like_lockfile_or_dependency_dump(example):
        return tier_from_signal(trash=True, reason="lockfile_or_dependency_dump")
    if looks_minified(text):
        return tier_from_signal(trash=True, reason="minified_or_raw_dump")
    if mostly_license_boilerplate(text):
        return tier_from_signal(trash=True, reason="mostly_license_boilerplate")
    if has_repeated_character_spam(text) or has_low_entropy(text):
        return tier_from_signal(trash=True, reason="low_entropy_or_repeated_characters")
    if has_repetition_spam(text):
        return tier_from_signal(trash=True, reason="repetition_spam")
    if contains_dangerous_code(text):
        return tier_from_signal(trash=True, reason="dangerous_code")
    return None


def general_quality_filter(
    example: TrainingExample, *, deduper: ExactDeduper | None = None
) -> FilterDecision:
    trash = obvious_trash_filter(example, deduper=deduper)
    if trash is not None:
        return trash
    if estimate_token_count(example.text) > 8192:
        return tier_from_signal(rescue=True, reason="long_example_needs_chunking_or_inspection")
    if has_programming_signal(example.text):
        return tier_from_signal(high_signal=True, reason="programming_signal")
    return tier_from_signal(useful=True, reason="general_useful_text")
