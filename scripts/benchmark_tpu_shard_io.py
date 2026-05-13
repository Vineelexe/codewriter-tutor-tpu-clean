from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cw360.prepack.reader import PrepackedShardReader  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark prepacked shard batch reads.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--num-batches", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    reader = PrepackedShardReader(args.input_dir, split=args.split, mmap=True)
    started = time.perf_counter()
    batches = 0
    sequences = 0
    for batch in reader.iter_batches(batch_size=args.batch_size):
        batches += 1
        sequences += int(batch.input_ids.shape[0])
        if batches >= args.num_batches:
            break
    elapsed = max(time.perf_counter() - started, 1.0e-9)
    print(
        f"read {batches} batches / {sequences} sequences in {elapsed:.4f}s "
        f"({sequences / elapsed:.2f} seq/s)"
    )


if __name__ == "__main__":
    main()
