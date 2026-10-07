from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def outcome(records: list[dict], threshold: float) -> dict:
    answerable = [record for record in records if record["answerable"]]
    unanswerable = [record for record in records if not record["answerable"]]
    answered = lambda record: record["status"] == "answered" and record["top_rerank_score"] >= threshold
    kept = sum(1 for record in answerable if answered(record)) / len(answerable)
    rejected = sum(1 for record in unanswerable if not answered(record)) / len(unanswerable)
    correct_context = sum(1 for record in answerable if answered(record) and record["context_hit"]) / len(answerable)
    return {
        "threshold": threshold,
        "balanced_accuracy": round((kept + rejected) / 2, 4),
        "answerable_answered": round(kept, 4),
        "answerable_answered_with_relevant_context": round(correct_context, 4),
        "unanswerable_not_found": round(rejected, 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Choose the reranker abstention threshold end to end on ungated validation runs.")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    records = [json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(not record["id"].startswith("validation_") for record in records):
        print("calibration must use validation records only", file=sys.stderr)
        return 1
    if all(record["answerable"] for record in records) or not any(record["answerable"] for record in records):
        print("calibration needs both answerable and unanswerable validation records", file=sys.stderr)
        return 1
    scores = sorted({record["top_rerank_score"] for record in records})
    thresholds = [0.0, *(round((low + high) / 2, 6) for low, high in zip(scores, scores[1:]))]
    table = [outcome(records, threshold) for threshold in thresholds]
    best = max(table, key=lambda row: (row["balanced_accuracy"], -row["threshold"]))
    reference = [outcome(records, threshold) for threshold in (0.0, 0.5, 0.9, 0.98)]
    args.output.write_text(
        json.dumps(
            {
                "criterion": "max balanced accuracy of final answered/NOT_FOUND decision on validation; candidate thresholds are midpoints between observed top scores; lowest threshold on ties",
                "selected": best,
                "reference_points": reference,
                "table": table,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(best["threshold"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
