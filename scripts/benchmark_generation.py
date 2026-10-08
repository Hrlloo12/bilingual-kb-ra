from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from rag.config import REPO_ROOT, load_serving_config
from rag.evaluation.generation import Judge, fact_coverage, precision, rouge_l
from rag.evaluation.relevance import chunk_fact_labels
from rag.grounding import strip_markers
from rag.ingestion import CHUNKS_FILE_NAME, MANIFEST_NAME, read_chunks

JUDGE_MODEL = "Qwen/Qwen3-8B-FP8"
RELEVANCE_SCORE = {"full": 1.0, "partial": 0.5, "none": 0.0}
CROSS_LINGUAL = {"ar_en", "en_ar"}


def mean(values: list[float]) -> float | None:
    values = [value for value in values if value is not None]
    return round(float(np.mean(values)), 4) if values else None


def summarize(rows: list[dict]) -> dict:
    import sacrebleu

    answerable = [row for row in rows if row["answerable"]]
    answered = [row for row in answerable if row["status"] == "answered"]
    unanswerable = [row for row in rows if not row["answerable"]]
    hypotheses = [strip_markers(row["answer"]) for row in answered]
    references = [row["reference_answer"] for row in answered]
    summary = {
        "queries": len(rows),
        "answerable": len(answerable),
        "answered": len(answered),
        "unanswerable": len(unanswerable),
        "judged": sum(1 for row in answered if row["judge"]),
        "judge_failures": sum(1 for row in answered if not row["judge"]),
        "faithfulness_supported_rate": mean([float(row["judge"]["supported"]) for row in answered if row["judge"]]),
        "hallucination_rate_answered": mean([float(not row["judge"]["supported"]) for row in answered if row["judge"]]),
        "answer_relevance": mean([RELEVANCE_SCORE[row["judge"]["relevance"]] for row in answered if row["judge"]]),
        "unanswerable_answered_rate": mean([float(row["status"] == "answered") for row in unanswerable]),
        "not_found_accuracy_unanswerable": mean([float(row["status"] == "not_found") for row in unanswerable]),
        "false_not_found_rate": mean([float(row["status"] == "not_found") for row in answerable]),
        "citation_precision": mean([row["citation_precision"] for row in answered]),
        "citation_recall_facts": mean([row["citation_recall"] for row in answered]),
        "context_precision": mean([row["context_precision"] for row in answerable]),
        "context_recall_facts": mean([row["context_recall"] for row in answerable]),
        "language_correct": mean([float(bool(row["post_checks"].get("language_ok"))) for row in answered]),
        "rouge_l": mean([row["rouge_l"] for row in answered]),
    }
    if answered:
        summary["chrf"] = round(sacrebleu.corpus_chrf(hypotheses, [references]).score, 2)
        summary["bleu"] = round(sacrebleu.corpus_bleu(hypotheses, [references], tokenize="intl").score, 2)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score Smart AI Search answers: faithfulness, citations, context, overlap and language.")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    args = parser.parse_args(argv)

    config = load_serving_config()
    chunks = {chunk.chunk_id: chunk for chunk in read_chunks(config.paths.corpus_processed / CHUNKS_FILE_NAME)}
    labels = chunk_fact_labels(list(chunks.values()), config.paths.corpus_raw / MANIFEST_NAME)
    facts_of = {
        row["id"]: set(row["relevant_fact_ids"])
        for row in map(json.loads, (REPO_ROOT / "data" / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines())
    }
    rows = [json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line.strip()]

    judge = Judge(args.judge_url, args.judge_model)
    for row in rows:
        targets = facts_of[row["id"]]
        cited = row["cited_chunk_ids"]
        answered = row["status"] == "answered"
        row["citation_precision"] = precision(cited, labels, targets) if answered else None
        row["citation_recall"] = fact_coverage(cited, labels, targets) if answered and targets else None
        row["context_precision"] = precision(row["context_chunk_ids"], labels, targets) if targets else None
        row["context_recall"] = fact_coverage(row["context_chunk_ids"], labels, targets) if targets else None
        row["judge"], row["rouge_l"] = None, None
        if not answered:
            continue
        passages = [f"{chunks[chunk_id].context_header}\n{chunks[chunk_id].text}" for chunk_id in (cited or row["context_chunk_ids"])]
        row["judge"] = judge.judge(row["query"], passages, row["answer"])
        if row["answerable"]:
            row["rouge_l"] = round(rouge_l(row["answer"], row["reference_answer"]), 4)

    groups: dict[str, dict[str, list]] = {"bucket": defaultdict(list), "language": defaultdict(list)}
    for row in rows:
        groups["bucket"][row["bucket"]].append(row)
        groups["language"][row["language"]].append(row)
    report = {
        "records": str(args.records),
        "judge_model": args.judge_model,
        "notes": {
            "faithfulness_supported_rate": "share of answered questions whose every claim the judge finds supported by the cited passages",
            "answer_relevance": "judge relevance: full=1, partial=0.5, none=0",
            "citation_recall_facts": "share of the query's gold facts stated by the cited chunks",
            "context_recall_facts": "share of the query's gold facts stated by the passages given to the generator",
        },
        "overall": summarize(rows),
        "cross_lingual": summarize([row for row in rows if row["bucket"] in CROSS_LINGUAL]),
        "by_bucket": {name: summarize(items) for name, items in sorted(groups["bucket"].items())},
        "by_language": {name: summarize(items) for name, items in sorted(groups["language"].items())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    scored = args.output.with_name(args.output.stem + "_records.jsonl")
    scored.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
