from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from rag.config import REPO_ROOT, load_serving_config
from rag.evaluation.relevance import chunk_fact_labels
from rag.facts import FactBase
from rag.grounding import strip_markers
from rag.ingestion import CHUNKS_FILE_NAME, MANIFEST_NAME, read_chunks

EXAMPLES = 6
NEGATION = re.compile(r"\b(not|no|never|without|isn't|aren't|doesn't|don't|can't|cannot)\b|(?:^|\s)(لا|ليس|ليست|غير|بدون|ما\s+[^\s]+ش|مو|مب)(?:\s|$)", re.IGNORECASE)
PRICE = re.compile(r"ريال|SAR|price|سعر|cost|تكلفة|رسوم|fee", re.IGNORECASE)
DATE = re.compile(r"\b(19|20)\d{2}\b|يناير|فبراير|مارس|أبريل|مايو|يونيو|يوليو|أغسطس|سبتمبر|أكتوبر|نوفمبر|ديسمبر|January|February|March|April|May|June|July|August|September|October|November|December|\d{1,2}:\d{2}|ساعة|hour|day|يوم|أيام|week|أسبوع", re.IGNORECASE)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def example(row: dict, reason: str) -> dict:
    return {
        "id": row["id"],
        "bucket": row.get("bucket") or row.get("languages"),
        "query": row["query"],
        "answer": strip_markers(row.get("answer", "")) if row.get("status", "answered") == "answered" else "NOT_FOUND",
        "reference": row.get("reference_answer", ""),
        "reason": reason,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect categorized error examples from final evaluation records.")
    parser.add_argument("--smart-records", type=Path, required=True, help="benchmark_generation *_records.jsonl")
    parser.add_argument("--interactive-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "error_analysis" / "error_examples.json")
    args = parser.parse_args(argv)

    config = load_serving_config()
    fact_base = FactBase.load(config.paths.facts_dir)
    chunks = read_chunks(config.paths.corpus_processed / CHUNKS_FILE_NAME)
    labels = chunk_fact_labels(chunks, config.paths.corpus_raw / MANIFEST_NAME)
    facts_of = {row["id"]: set(row["relevant_fact_ids"]) for row in read_jsonl(REPO_ROOT / "data" / "test.jsonl")}
    smart = read_jsonl(args.smart_records)
    interactive = read_jsonl(args.interactive_records)

    categories: dict[str, list[dict]] = {name: [] for name in (
        "retrieval_errors", "hallucination", "partial_or_wrong_attribute", "citation_errors", "language_drift", "mixed_language_issues", "memory_errors",
        "incorrect_numbers", "incorrect_dates_or_durations", "incorrect_prices", "names_entities", "negation",
        "not_found_false_refusal", "not_found_missed", "ar_to_en_weakness",
    )}
    for row in smart:
        targets = facts_of.get(row["id"], set())
        answered = row["status"] == "answered"
        if row["answerable"] and not row["context_hit"]:
            categories["retrieval_errors"].append(example(row, "no passage stating the target fact reached the generator"))
        unsupported = answered and bool(row.get("judge")) and not row["judge"]["supported"]
        if unsupported:
            categories["hallucination"].append(example(row, f"judge: unsupported claims {row['judge']['unsupported_claims']}"))
        if answered and row.get("judge") and row["judge"]["relevance"] != "full":
            categories["partial_or_wrong_attribute"].append(example(row, f"judge relevance: {row['judge']['relevance']}"))
        if answered and not row["answerable"]:
            categories["hallucination"].append(example(row, "unanswerable question answered"))
            categories["not_found_missed"].append(example(row, "unanswerable question answered"))
        if answered and row["answerable"] and row["cited_chunk_ids"] and not any(labels.get(chunk_id, frozenset()) & targets for chunk_id in row["cited_chunk_ids"]):
            confusable = set().union(*(fact_base.confusables(fact_id) for fact_id in targets)) if targets else set()
            cited_facts = set().union(*(labels.get(chunk_id, frozenset()) for chunk_id in row["cited_chunk_ids"]))
            categories["citation_errors"].append(example(row, f"cited {row['cited_chunk_ids']} which do not state the target fact"))
            if cited_facts & confusable:
                categories["names_entities"].append(example(row, f"answered from a confusable entity: {sorted(cited_facts & confusable)}"))
        if answered and not row["post_checks"].get("language_ok", True):
            categories["language_drift"].append(example(row, "answer language does not match the question"))
        if row["bucket"] == "mixed" and (not answered or not row["context_hit"] or not row["post_checks"].get("language_ok", True)):
            categories["mixed_language_issues"].append(example(row, f"status={row['status']} context_hit={row['context_hit']}"))
        if answered and row["answerable"] and row["reference_numbers"] and not set(row["reference_numbers"]) <= set(row["answer_numbers"]):
            missing = sorted(set(row["reference_numbers"]) - set(row["answer_numbers"]))
            reason = f"reference numbers {missing} missing from answer"
            if PRICE.search(row["reference_answer"]):
                categories["incorrect_prices"].append(example(row, reason))
            elif DATE.search(row["reference_answer"]):
                categories["incorrect_dates_or_durations"].append(example(row, reason))
            else:
                categories["incorrect_numbers"].append(example(row, reason))
        if row["post_checks"].get("unsupported_codes"):
            categories["names_entities"].append(example(row, f"unsupported codes {row['post_checks']['unsupported_codes']}"))
        if NEGATION.search(row["query"]) and (row["answerable"] != answered or unsupported):
            categories["negation"].append(example(row, f"negated question, status={row['status']}"))
        elif answered and row["answerable"] and NEGATION.search(strip_markers(row["answer"])) and not NEGATION.search(row["reference_answer"]):
            categories["negation"].append(example(row, "answer negates a fact the reference states positively"))
        if row["answerable"] and not answered:
            categories["not_found_false_refusal"].append(example(row, f"{row['abstain_reason']} (top reranker score {row['top_rerank_score']})"))
        if row["bucket"] == "ar_en" and (not answered or not row["context_hit"] or unsupported):
            categories["ar_to_en_weakness"].append(example(row, f"status={row['status']} context_hit={row['context_hit']}"))

    for row in interactive:
        result = row["interactive"]
        if row["needs_rewrite"] and result["rank"] != 1 and (row.get("oracle") or {}).get("rank") == 1:
            categories["memory_errors"].append(
                {
                    "id": row["id"], "bucket": row["languages"], "query": f"{row['first_query']} → {row['query']}",
                    "answer": row["rewrite"]["standalone_query"], "reference": row["gold_standalone"],
                    "reason": f"rewrite lost context: rank {result['rank']} vs oracle rank 1",
                }
            )
        if not row["needs_rewrite"] and row["rewrite"]["applied"]:
            categories["memory_errors"].append(
                {"id": row["id"], "bucket": row["languages"], "query": row["query"], "answer": row["rewrite"]["standalone_query"], "reference": row["query"], "reason": "standalone question rewritten (gate false positive)"}
            )

    report = {
        "sources": {"smart": str(args.smart_records), "interactive": str(args.interactive_records)},
        "counts": {name: len(items) for name, items in categories.items()},
        "examples": {name: items[:EXAMPLES] for name, items in categories.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["counts"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
