from __future__ import annotations

import hashlib

from cw360.data.schemas import TrainingExample


def deterministic_split(example: TrainingExample, *, val_pct: int = 2, test_pct: int = 0) -> str:
    if val_pct < 0 or test_pct < 0 or val_pct + test_pct >= 100:
        raise ValueError("val_pct and test_pct must be non-negative and sum to less than 100")
    key = f"{example.source}:{example.metadata.get('id', '')}:{example.text[:200]}"
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < test_pct:
        return "test"
    if bucket < test_pct + val_pct:
        return "validation"
    return "train"
