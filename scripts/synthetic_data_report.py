from __future__ import annotations

import argparse
import json

from cw360.synthetic.dedupe import dedupe_records
from cw360.synthetic.reports import build_report
from cw360.synthetic.validate import iter_validation_results, valid_records_from_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a structured synthetic data report.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--allow-duplicates", action="store_true")
    args = parser.parse_args()

    results = list(iter_validation_results(args.input))
    valid_records = valid_records_from_results(results)
    deduped = dedupe_records(valid_records, allow_duplicates=args.allow_duplicates)
    report = build_report(results, deduped.duplicates).to_dict()
    report["valid_after_dedupe"] = len(deduped.records)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
