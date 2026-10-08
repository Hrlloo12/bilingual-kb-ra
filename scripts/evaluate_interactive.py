from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import httpx
import numpy as np

from rag.config import REPO_ROOT, load_serving_config
from rag.evaluation.answers import number_set
from rag.evaluation.metrics import latency_summary
from rag.evaluation.relevance import chunk_fact_labels
from rag.ingestion import CHUNKS_FILE_NAME, MANIFEST_NAME, read_chunks
from rag.langdetect import detect_language

SYSTEMS = ("interactive", "no_rewrite", "oracle")


def first_rank(retrieved: list[dict], relevant: set[str]) -> int | None:
    return next((rank for rank, passage in enumerate(retrieved, start=1) if passage["chunk_id"] in relevant), None)


def outcome(result: dict, relevant: set[str], reference_numbers: list[float]) -> dict:
    rank = first_rank(result["retrieved"], relevant)
    answered = result["status"] == "answered"
    return {
        "status": result["status"],
        "answer": result["answer"],
        "rank": rank,
        "cited_relevant": any(citation["chunk_id"] in relevant for citation in result["citations"]),
        "numbers_ok": answered and bool(reference_numbers) and set(reference_numbers) <= set(number_set(result["answer"])),
        "language_ok": bool(result["post_checks"].get("language_ok")) if answered else None,
        "total_ms": result["latency_ms"]["total"],
    }


def language_preserved(standalone: str, language: str) -> bool:
    detected = detect_language(standalone)
    return detected == "en" if language == "en" else detected in ("ar", "mixed")


def rate(values: list[bool]) -> float | None:
    return round(float(np.mean(values)), 4) if values else None


def summarize(records: list[dict]) -> dict:
    follow_ups = [record for record in records if record["needs_rewrite"]]
    standalone = [record for record in records if not record["needs_rewrite"]]
    gate_true = sum(1 for record in follow_ups if record["rewrite"]["applied"])
    gate_false_positive = sum(1 for record in standalone if record["rewrite"]["applied"])
    applied = [record for record in records if record["rewrite"]["applied"]]
    summary = {
        "conversations": len(records),
        "gate": {
            "follow_ups": len(follow_ups),
            "standalone": len(standalone),
            "follow_up_recall": rate([record["rewrite"]["applied"] for record in follow_ups]),
            "standalone_rewritten_rate": rate([record["rewrite"]["applied"] for record in standalone]),
            "precision": round(gate_true / (gate_true + gate_false_positive), 4) if gate_true + gate_false_positive else None,
        },
        "rewrite": {
            "applied": len(applied),
            "fallbacks": sum(1 for record in applied if record["rewrite"]["fallback"]),
            "language_preserved": rate([language_preserved(record["rewrite"]["standalone_query"], record["language"]) for record in applied]),
            "latency_ms": latency_summary([record["rewrite"]["latency_ms"] for record in applied]) if applied else {},
            "skip_path_overhead_ms": latency_summary([record["interactive_latency"]["memory_read"] + record["interactive_latency"]["memory_write"] for record in records]),
        },
        "interactive_total_ms": latency_summary([record["interactive_latency"]["total"] for record in records]),
        "suggested_followups": followup_summary(records),
        "systems": {},
    }
    for system in SYSTEMS:
        rows = [record[system] for record in records if record.get(system)]
        if not rows:
            continue
        ranks = [row["rank"] for row in rows]
        summary["systems"][system] = {
            "n": len(rows),
            "hit@1": rate([rank == 1 for rank in ranks]),
            "hit@5": rate([rank is not None and rank <= 5 for rank in ranks]),
            "mrr@10": round(float(np.mean([1 / rank if rank else 0.0 for rank in ranks])), 4),
            "answered": rate([row["status"] == "answered" for row in rows]),
            "cited_relevant": rate([row["cited_relevant"] for row in rows]),
            "numbers_ok": rate([row["numbers_ok"] for row in rows if row["status"] == "answered"]),
            "language_ok": rate([row["language_ok"] for row in rows if row["language_ok"] is not None]),
        }
    return summary


def followup_summary(records: list[dict]) -> dict:
    rows = [record for record in records if "suggested_followups" in record]
    if not rows:
        return {}
    return {
        "turns": len(rows),
        "with_suggestions": rate([bool(record["suggested_followups"]) for record in rows]),
        "average_count": round(float(np.mean([len(record["suggested_followups"]) for record in rows])), 2),
        "in_user_language": rate(
            [language_preserved(question, record["language"]) for record in rows for question in record["suggested_followups"]]
        ),
    }


def grouped(records: list[dict], key: str) -> dict:
    groups = defaultdict(list)
    for record in records:
        groups[record[key]].append(record)
    return {name: summarize(rows) for name, rows in sorted(groups.items())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Interactive AI Search through the API on multi-turn conversations.")
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--label", default="interactive")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "interactive")
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    args = parser.parse_args(argv)

    config = load_serving_config()
    chunks = read_chunks(config.paths.corpus_processed / CHUNKS_FILE_NAME)
    labels = chunk_fact_labels(chunks, config.paths.corpus_raw / MANIFEST_NAME)
    client = httpx.Client(base_url=args.base_url, timeout=120)
    output_dir = args.output_dir / args.label
    output_dir.mkdir(parents=True, exist_ok=True)

    def call(body: dict) -> dict:
        response = client.post("/v1/search", json=body)
        response.raise_for_status()
        return response.json()

    report = {"label": args.label, "base_url": args.base_url, "splits": {}}
    for split in args.splits:
        path = REPO_ROOT / "data" / "interactive" / f"conversations_{split}.jsonl"
        conversations = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()][: args.limit]
        records = []
        for conversation in conversations:
            first, second = conversation["turns"]
            relevant = {chunk_id for chunk_id, facts in labels.items() if facts & set(second["relevant_fact_ids"])}
            reference_numbers = number_set(second["reference_answer"])
            opening = call({"mode": "interactive", "query": first["query"]})
            follow_up = call({"mode": "interactive", "query": second["query"], "session_id": opening["session_id"]})
            record = {
                "id": conversation["id"],
                "type": conversation["type"],
                "languages": conversation["languages"],
                "language": second["language"],
                "needs_rewrite": second["needs_rewrite"],
                "first_query": first["query"],
                "query": second["query"],
                "gold_standalone": second["gold_standalone"],
                "rewrite": {**follow_up["rewrite"], "standalone_query": follow_up["rewritten_query"]},
                "suggested_followups": follow_up["suggested_followups"],
                "interactive_latency": follow_up["latency_ms"],
                "interactive": outcome(follow_up, relevant, reference_numbers),
            }
            if "no_rewrite" in args.systems:
                record["no_rewrite"] = outcome(call({"mode": "smart_ai_search", "query": second["query"]}), relevant, reference_numbers)
            if "oracle" in args.systems and second["needs_rewrite"]:
                record["oracle"] = outcome(call({"mode": "smart_ai_search", "query": second["gold_standalone"]}), relevant, reference_numbers)
            client.delete(f"/v1/sessions/{opening['session_id']}")
            records.append(record)
        (output_dir / f"records_{split}.jsonl").write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
        report["splits"][split] = {"overall": summarize(records), "by_type": grouped(records, "type"), "by_languages": grouped(records, "languages")}
        print(json.dumps({split: report["splits"][split]["overall"]}, indent=2, ensure_ascii=False))
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
