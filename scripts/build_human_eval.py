from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from rag.config import REPO_ROOT, load_serving_config
from rag.grounding import strip_markers
from rag.ingestion import CHUNKS_FILE_NAME, read_chunks

GROUPS = {
    "arabic": {"ar_ar", "dialect"},
    "english": {"en_en"},
    "mixed_or_cross_lingual": {"mixed", "ar_en", "en_ar"},
}
PER_GROUP = 10
UNANSWERABLE_PER_GROUP = 1
RATING_COLUMNS = ["correctness_1_to_5", "faithful_to_sources_1_to_5", "citations_correct_yes_no", "language_correct_yes_no", "fluency_1_to_5", "notes"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sample 30 Smart AI Search answers for human rating (ratings are left empty).")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "human_eval" / "human_eval_sheet.csv")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)

    config = load_serving_config()
    chunks = {chunk.chunk_id: chunk for chunk in read_chunks(config.paths.corpus_processed / CHUNKS_FILE_NAME)}
    rows = [json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = random.Random(args.seed)
    unanswerable = [row for row in rows if not row["answerable"]]
    rng.shuffle(unanswerable)

    selected = []
    for group, buckets in GROUPS.items():
        pool = [row for row in rows if row["answerable"] and row["bucket"] in buckets]
        rng.shuffle(pool)
        language = "en" if group == "english" else "ar"
        extra = [row for row in unanswerable if row["language"] == language and row not in [item for _, item in selected]][:UNANSWERABLE_PER_GROUP]
        for row in pool[: PER_GROUP - len(extra)] + extra:
            selected.append((group, row))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item", "id", "group", "bucket", "question", "system_answer", "status", "sources", "source_text", "reference_answer", *RATING_COLUMNS])
        for item, (group, row) in enumerate(selected, start=1):
            cited = [chunks[chunk_id] for chunk_id in row["cited_chunk_ids"] if chunk_id in chunks]
            sources = " | ".join(f"{chunk.title} › {chunk.section or ''} ({chunk.source})" for chunk in cited)
            source_text = "\n---\n".join(chunk.text for chunk in cited)
            answer = strip_markers(row["answer"]) if row["status"] == "answered" else "NOT_FOUND"
            writer.writerow([item, row["id"], group, row["bucket"], row["query"], answer, row["status"], sources, source_text, row["reference_answer"], *[""] * len(RATING_COLUMNS)])
    print(f"wrote {len(selected)} items to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
