from __future__ import annotations

import argparse
import json
from pathlib import Path

from cw360.synthetic.dedupe import dedupe_records
from cw360.synthetic.reports import build_report
from cw360.synthetic.validate import iter_validation_results, valid_records_from_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate structured synthetic JSONL data.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--allow-duplicates", action="store_true")
    args = parser.parse_args()

    results = list(iter_validation_results(args.input))
    valid_records = valid_records_from_results(results)
    deduped = dedupe_records(valid_records, allow_duplicates=args.allow_duplicates)
    report = build_report(results, deduped.duplicates).to_dict()
    report["valid_after_dedupe"] = len(deduped.records)

    output_path = Path(args.output_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"validated {report['total_records']} records: "
        f"{report['valid_after_dedupe']} valid after dedupe, "
        f"{report['rejected_records']} rejected, "
        f"{report['duplicate_records']} duplicates"
    )


if __name__ == "__main__":
    main()
