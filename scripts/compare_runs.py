"""Collect actual evaluation reports into a flat comparison CSV."""

import argparse
import csv
import json
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--output", default="outputs/comparison.csv")
    args = parser.parse_args()
    rows = []
    for filename in args.reports:
        report = json.loads(Path(filename).read_text())
        rows.append(
            {"report": filename, **{k: v for k, v in report.items() if not isinstance(v, (dict, list))}}
        )
    columns = sorted(set().union(*(row.keys() for row in rows)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(output)
