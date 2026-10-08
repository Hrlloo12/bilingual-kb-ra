from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from itertools import zip_longest
from pathlib import Path

from rag.config import REPO_ROOT, load_serving_config
from rag.grounding import strip_markers
from rag.ingestion import CHUNKS_FILE_NAME, read_chunks
from rag.langdetect import detect_language

GROUPS = {
    "arabic": {"ar_ar", "dialect"},
    "english": {"en_en"},
    "mixed_or_cross_lingual": {"mixed", "ar_en", "en_ar"},
}
PER_GROUP = 10
UNANSWERABLE_PER_GROUP = 1
RATING_COLUMNS = [
    "meaning_accuracy_1_to_5",
    "faithfulness_1_to_5",
    "fluency_1_to_5",
    "completeness_1_to_5",
    "citation_quality_1_to_5",
    "language_correct_yes_no",
    "notes",
]
COLUMNS = ["item", "id", "group", "bucket", "question", "answer_language", "system_answer", "status", "sources", "source_text", "reference_answer", *RATING_COLUMNS]


def sample(rows: list[dict], seed: int) -> list[tuple[str, dict]]:
    rng = random.Random(seed)
    unanswerable = [row for row in rows if not row["answerable"]]
    rng.shuffle(unanswerable)
    selected: list[tuple[str, dict]] = []
    for group, buckets in GROUPS.items():
        by_bucket = {bucket: [row for row in rows if row["answerable"] and row["bucket"] == bucket] for bucket in sorted(buckets)}
        for pool in by_bucket.values():
            rng.shuffle(pool)
        interleaved = [row for batch in zip_longest(*by_bucket.values()) for row in batch if row is not None]
        language = "en" if group == "english" else "ar"
        taken = {item["id"] for _, item in selected}
        extra = [row for row in unanswerable if row["language"] == language and row["id"] not in taken][:UNANSWERABLE_PER_GROUP]
        selected.extend((group, row) for row in interleaved[: PER_GROUP - len(extra)] + extra)
    return selected


def same_items(rows: list[dict], sheet: Path) -> list[tuple[str, dict]]:
    by_id = {row["id"]: row for row in rows}
    with sheet.open(encoding="utf-8-sig", newline="") as handle:
        return [(item["group"], by_id[item["id"]]) for item in csv.DictReader(handle)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sample 30 Smart AI Search answers (10 Arabic, 10 English, 10 mixed or cross-lingual) for human rating.")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "human_eval" / "human_eval_sheet.csv")
    parser.add_argument("--same-items-as", type=Path, default=None, help="Reuse the questions of an existing sheet, for example to rate the base models on the same items.")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)

    config = load_serving_config()
    chunks = {chunk.chunk_id: chunk for chunk in read_chunks(config.paths.corpus_processed / CHUNKS_FILE_NAME)}
    rows = [json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = same_items(rows, args.same_items_as) if args.same_items_as else sample(rows, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for item, (group, row) in enumerate(selected, start=1):
            cited = [chunks[chunk_id] for chunk_id in row["cited_chunk_ids"] if chunk_id in chunks]
            sources = " | ".join(f"{chunk.title} › {chunk.section or ''} ({chunk.source})" for chunk in cited)
            source_text = "\n---\n".join(chunk.text for chunk in cited)
            answered = row["status"] == "answered"
            answer = strip_markers(row["answer"]) if answered else "NOT_FOUND"
            answer_language = ("en" if detect_language(answer) == "en" else "ar") if answered else ""
            writer.writerow(
                [item, row["id"], group, row["bucket"], row["query"], answer_language, answer, row["status"], sources, source_text, row["reference_answer"], *[""] * len(RATING_COLUMNS)]
            )
    print(f"wrote {len(selected)} items to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
