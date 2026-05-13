from __future__ import annotations

import argparse
from pathlib import Path

from cw360.synthetic.dedupe import dedupe_records
from cw360.synthetic.local_writer import write_records_jsonl
from cw360.synthetic.splits import SplitFractions, split_records
from cw360.synthetic.validate import iter_validation_results, valid_records_from_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate, dedupe, and split synthetic JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.98)
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--test-fraction", type=float, default=0.01)
    parser.add_argument("--allow-duplicates", action="store_true")
    args = parser.parse_args()

    results = list(iter_validation_results(args.input))
    valid_records = valid_records_from_results(results)
    deduped = dedupe_records(valid_records, allow_duplicates=args.allow_duplicates)
    fractions = SplitFractions(args.train_fraction, args.val_fraction, args.test_fraction)
    splits = split_records(deduped.records, fractions=fractions)
    output_dir = Path(args.output_dir)
    for split, records in splits.items():
        write_records_jsonl(records, output_dir / f"{split}.jsonl")
    print(
        "wrote synthetic splits "
        + ", ".join(f"{split}={len(records)}" for split, records in splits.items())
    )


if __name__ == "__main__":
    main()
