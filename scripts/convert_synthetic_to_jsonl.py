from __future__ import annotations

import argparse

from cw360.synthetic.convert import record_to_jsonl_row
from cw360.synthetic.dedupe import dedupe_records
from cw360.synthetic.local_writer import write_rows_jsonl
from cw360.synthetic.validate import iter_validation_results, valid_records_from_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert validated structured synthetic records to prepack-ready JSONL."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-duplicates", action="store_true")
    args = parser.parse_args()

    results = list(iter_validation_results(args.input))
    valid_records = valid_records_from_results(results)
    deduped = dedupe_records(valid_records, allow_duplicates=args.allow_duplicates)
    rows = [record_to_jsonl_row(record) for record in deduped.records]
    write_rows_jsonl(rows, args.output)
    print(f"wrote {len(rows)} prepack-ready synthetic examples to {args.output}")


if __name__ == "__main__":
    main()
