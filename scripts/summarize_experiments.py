#!/usr/bin/env python3
"""Create paper-ready metric tables from frozen report JSON files.

Only report metrics belong in the main paper table. This tool deliberately
rejects dev metrics so it cannot turn model-selection measurements into claims.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = ("mAP", "R@1", "R@5", "R@10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.labels and len(args.labels) != len(args.metrics):
        raise SystemExit("--labels must have zero entries or one per --metrics file")
    labels = args.labels or [path.parent.name for path in args.metrics]
    rows = []
    for label, path in zip(labels, args.metrics):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("split") != "report":
            raise ValueError(f"{path} is split={payload.get('split')!r}; only report metrics are allowed")
        missing = [key for key in METRICS if key not in payload]
        if missing:
            raise ValueError(f"{path} lacks metrics: {missing}")
        rows.append({"method": label, **{key: float(payload[key]) for key in METRICS}})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "report_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("method", *METRICS))
        writer.writeheader()
        writer.writerows(rows)
    tex_path = args.output_dir / "report_metrics.tex"
    row_end = r" \\"
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Method & mAP & R@1 & R@5 & R@10" + row_end,
        "\\midrule",
    ]
    lines.extend(
        f"{row['method']} & " + " & ".join(f"{100 * row[key]:.2f}" for key in METRICS) + row_end
        for row in rows
    )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote: {csv_path}\nwrote: {tex_path}")


if __name__ == "__main__":
    main()
