from __future__ import annotations

import os
import random


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required to seed NumPy RNG state") from exc
    np.random.seed(seed)
