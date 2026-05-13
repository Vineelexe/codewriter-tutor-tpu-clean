from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cw360.prepack.reader import PrepackedShardReader  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview token IDs from TPU prepacked shards.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    reader = PrepackedShardReader(args.input_dir, split=args.split, mmap=True)
    for index, (sequence, cursor) in enumerate(reader.iter_sequences()):
        if index >= args.limit:
            break
        preview = [int(token_id) for token_id in sequence[:12]]
        print(
            f"{args.split}[{index}] shape={tuple(sequence.shape)} "
            f"first_tokens={preview} next_cursor={cursor.to_dict()}"
        )


if __name__ == "__main__":
    main()
