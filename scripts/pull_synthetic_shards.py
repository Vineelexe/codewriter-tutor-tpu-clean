from __future__ import annotations

import argparse
import os
from itertools import islice

from cw360.synthetic.remote_read import iter_hf_dataset_rows, pull_synthetic_rows_to_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pull structured synthetic shards from a Hugging Face Dataset repo."
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = iter_hf_dataset_rows(
        args.repo_id,
        split=args.split,
        streaming=True,
        token=os.environ.get("HF_TOKEN") or None,
    )
    if args.max_records is not None:
        rows = islice(rows, args.max_records)
    written = pull_synthetic_rows_to_dir(
        rows,
        args.output_dir,
        shard_size=args.shard_size,
        overwrite=args.overwrite,
    )
    print(f"pulled {len(written)} synthetic shard file(s) into {args.output_dir}")


if __name__ == "__main__":
    main()
