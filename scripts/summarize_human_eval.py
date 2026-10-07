from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from rag.config import REPO_ROOT

SCALES = ("correctness_1_to_5", "faithful_to_sources_1_to_5", "fluency_1_to_5")
FLAGS = ("citations_correct_yes_no", "language_correct_yes_no")


def summarize(rows: list[dict]) -> dict:
    result = {"items": len(rows)}
    for column in SCALES:
        values = [float(row[column]) for row in rows if row[column].strip()]
        result[column] = {"rated": len(values), "mean": round(sum(values) / len(values), 2) if values else None}
    for column in FLAGS:
        values = [row[column].strip().lower() for row in rows if row[column].strip()]
        result[column] = {"rated": len(values), "yes_rate": round(sum(value.startswith("y") for value in values) / len(values), 3) if values else None}
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize the human ratings filled in by the reviewer.")
    parser.add_argument("--sheet", type=Path, default=REPO_ROOT / "results" / "human_eval" / "human_eval_sheet.csv")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "human_eval" / "human_eval_summary.json")
    args = parser.parse_args(argv)
    with args.sheet.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    unrated = [row["item"] for row in rows if not any(row[column].strip() for column in (*SCALES, *FLAGS))]
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["group"]].append(row)
    report = {"overall": summarize(rows), "by_group": {name: summarize(items) for name, items in sorted(by_group.items())}, "unrated_items": unrated}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
