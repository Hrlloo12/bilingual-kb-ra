from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag.evaluation.answers import number_set, summarize_by_bucket


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recompute derived fields and summaries of a Smart AI Search run from its per-query records.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args(argv)

    summary_path = args.run_dir / f"{args.label}_summary.json"
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    for split in list(report["splits"]):
        path = args.run_dir / f"{args.label}_{split}.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for record in records:
            reference = "" if record["reference_answer"] == "NOT_FOUND" else record["reference_answer"]
            record["reference_numbers"] = number_set(reference)
            record["answer_numbers"] = number_set(record["answer"]) if record["status"] == "answered" else []
        path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
        report["splits"][split] = summarize_by_bucket(records)
        overall = report["splits"][split]["overall"]
        print(split, {key: overall[key] for key in ("reference_numbers_all_in_answer_rate", "reference_numbers_n", "answer_numbers_all_in_reference_rate", "answer_numbers_n")})
    summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
