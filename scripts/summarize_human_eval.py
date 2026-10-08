from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from rag.config import REPO_ROOT

SCALES = ("meaning_accuracy_1_to_5", "faithfulness_1_to_5", "fluency_1_to_5", "completeness_1_to_5", "citation_quality_1_to_5")
FLAGS = ("language_correct_yes_no",)
EXPECTED_GROUPS = {"arabic": 10, "english": 10, "mixed_or_cross_lingual": 10}


def score(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    number = int(float(value))
    if number != float(value) or not 1 <= number <= 5:
        raise ValueError(f"ratings must be whole numbers from 1 to 5, got {value!r}")
    return number


def mean(values: list[int]) -> float | None:
    return round(sum(values) / len(values), 2) if values else None


def summarize(rows: list[dict]) -> dict:
    result: dict = {"items": len(rows)}
    for column in SCALES:
        values = [value for value in (score(row[column]) for row in rows) if value is not None]
        result[column] = {"rated": len(values), "mean": mean(values)}
    for column in FLAGS:
        values = [row[column].strip().lower() for row in rows if row[column].strip()]
        result[column] = {"rated": len(values), "yes_rate": round(sum(value.startswith("y") for value in values) / len(values), 3) if values else None}
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize the human ratings filled in by the reviewer.")
    parser.add_argument("--sheet", type=Path, default=REPO_ROOT / "results" / "human_eval" / "human_eval_sheet.csv")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = args.output or args.sheet.with_name(args.sheet.stem.replace("_sheet", "") + "_summary.json")
    with args.sheet.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    counts = defaultdict(int)
    by_group = defaultdict(list)
    for row in rows:
        counts[row["group"]] += 1
        by_group[row["group"]].append(row)
    fluency = {
        language: mean([value for value in (score(row["fluency_1_to_5"]) for row in rows if row["answer_language"] == language) if value is not None])
        for language in ("ar", "en")
    }
    report = {
        "sheet": args.sheet.name,
        "composition_ok": dict(counts) == EXPECTED_GROUPS,
        "overall": summarize(rows),
        "fluency_by_answer_language": fluency,
        "by_group": {name: summarize(items) for name, items in sorted(by_group.items())},
        "unrated_items": [row["item"] for row in rows if not any(row[column].strip() for column in SCALES)],
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
