from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_for_dedupe(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.strip())


def exact_content_hash(text: str) -> str:
    return hashlib.sha256(normalize_for_dedupe(text).encode("utf-8")).hexdigest()


@dataclass(slots=True)
class ExactDeduper:
    seen_hashes: set[str] = field(default_factory=set)

    def is_duplicate(self, text: str) -> bool:
        digest = exact_content_hash(text)
        if digest in self.seen_hashes:
            return True
        self.seen_hashes.add(digest)
        return False


def token_shingles(text: str, *, width: int = 5) -> set[str]:
    tokens = normalize_for_dedupe(text).split()
    if len(tokens) <= width:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[index : index + width]) for index in range(len(tokens) - width + 1)}


def jaccard_similarity(left: str, right: str) -> float:
    left_shingles = token_shingles(left)
    right_shingles = token_shingles(right)
    if not left_shingles and not right_shingles:
        return 1.0
    if not left_shingles or not right_shingles:
        return 0.0
    return len(left_shingles & right_shingles) / len(left_shingles | right_shingles)
