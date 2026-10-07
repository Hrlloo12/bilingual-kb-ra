from __future__ import annotations

from collections import defaultdict

from rag.evaluation.metrics import aggregate, query_metrics
from rag.evaluation.relevance import RetrievalQuery, relevant_chunk_ids

CROSS_LINGUAL_BUCKETS = {"ar_en", "en_ar"}
SUMMARY_KEYS = ("recall@1", "recall@5", "recall@10", "recall@20", "mrr", "ndcg@10", "hit@5")


def evaluate(rankings: list[list[str]], queries: list[RetrievalQuery], labels: dict) -> dict:
    groups: dict[str, dict[str, list]] = {"bucket": defaultdict(list), "language": defaultdict(list)}
    overall, cross_lingual, per_query = [], [], []
    for query, ranked in zip(queries, rankings, strict=True):
        if not query.answerable:
            continue
        relevant = relevant_chunk_ids(query, labels)
        metrics = query_metrics(ranked, relevant)
        overall.append(metrics)
        groups["bucket"][query.bucket].append(metrics)
        groups["language"][query.language].append(metrics)
        if "cross_lingual" in query.tags or query.bucket in CROSS_LINGUAL_BUCKETS:
            cross_lingual.append(metrics)
        first_rank = next((rank for rank, chunk_id in enumerate(ranked, start=1) if chunk_id in relevant), None)
        per_query.append({"id": query.id, "bucket": query.bucket, "first_relevant_rank": first_rank, "top3": ranked[:3]})
    return {
        "answerable_queries": len(overall),
        "overall": aggregate(overall),
        "cross_lingual": {"n": len(cross_lingual), **aggregate(cross_lingual)},
        "by_bucket": {name: {"n": len(rows), **aggregate(rows)} for name, rows in sorted(groups["bucket"].items())},
        "by_language": {name: {"n": len(rows), **aggregate(rows)} for name, rows in sorted(groups["language"].items())},
        "per_query": per_query,
    }


def print_summary(name: str, result: dict, keys: tuple[str, ...] = SUMMARY_KEYS) -> None:
    print(f"\n== {name}" + (f"  latency_ms={result['latency_ms']}" if "latency_ms" in result else ""))
    print(f"{'group':14s} {'n':>4s} " + " ".join(f"{key:>9s}" for key in keys))
    rows = [("overall", result["answerable_queries"], result["overall"]), ("cross_lingual", result["cross_lingual"]["n"], result["cross_lingual"])]
    rows += [(bucket, values["n"], values) for bucket, values in result["by_bucket"].items()]
    for label, count, values in rows:
        if count:
            print(f"{label:14s} {count:4d} " + " ".join(f"{values[key]:9.3f}" for key in keys))
