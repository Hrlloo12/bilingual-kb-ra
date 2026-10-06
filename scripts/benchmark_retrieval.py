from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

from rag.config import REPO_ROOT, ServingConfig, load_serving_config
from rag.evaluation.metrics import aggregate, latency_summary, query_metrics
from rag.evaluation.relevance import RetrievalQuery, chunk_fact_labels, load_queries, relevant_chunk_ids
from rag.facts import FactBase
from rag.ingestion import CHUNKS_FILE_NAME, MANIFEST_NAME, read_chunks
from rag.normalize import normalize_for_dense

CROSS_LINGUAL_BUCKETS = {"ar_en", "en_ar"}
Retriever = Callable[[str, int], list[str]]


def bm25_retriever(config: ServingConfig) -> Retriever:
    from rag.retrieval.bm25 import BM25Index

    index = BM25Index(config.opensearch)
    return lambda query, top_k: [hit.chunk.chunk_id for hit in index.search(normalize_for_dense(query), top_k)]


def dense_retriever(config: ServingConfig) -> Retriever:
    from rag.retrieval.dense import DenseIndex, Embedder

    embedder = Embedder(config.embedding)
    index = DenseIndex(config.qdrant)
    return lambda query, top_k: [hit.chunk.chunk_id for hit in index.search(embedder.encode_queries([query])[0], top_k)]


RETRIEVERS: dict[str, Callable[[ServingConfig], Retriever]] = {"bm25": bm25_retriever, "dense": dense_retriever}


def evaluate(retriever: Retriever, queries: list[RetrievalQuery], labels: dict, top_k: int) -> dict:
    retriever(queries[0].query, top_k)
    groups: dict[str, dict[str, list]] = {"bucket": defaultdict(list), "language": defaultdict(list)}
    overall: list[dict] = []
    cross_lingual: list[dict] = []
    latencies: list[float] = []
    per_query = []
    for query in queries:
        started = perf_counter()
        ranked = retriever(query.query, top_k)
        latencies.append((perf_counter() - started) * 1000)
        if not query.answerable:
            per_query.append({"id": query.id, "bucket": query.bucket, "top3": ranked[:3]})
            continue
        relevant = relevant_chunk_ids(query, labels)
        metrics = query_metrics(ranked, relevant)
        overall.append(metrics)
        groups["bucket"][query.bucket].append(metrics)
        groups["language"][query.language].append(metrics)
        if query.bucket in CROSS_LINGUAL_BUCKETS:
            cross_lingual.append(metrics)
        first_rank = next((rank for rank, chunk_id in enumerate(ranked, start=1) if chunk_id in relevant), None)
        per_query.append(
            {"id": query.id, "bucket": query.bucket, "first_relevant_rank": first_rank, "relevant": len(relevant), "top3": ranked[:3]}
        )
    return {
        "answerable_queries": len(overall),
        "overall": aggregate(overall),
        "cross_lingual": aggregate(cross_lingual),
        "by_bucket": {name: {"n": len(rows), **aggregate(rows)} for name, rows in sorted(groups["bucket"].items())},
        "by_language": {name: {"n": len(rows), **aggregate(rows)} for name, rows in sorted(groups["language"].items())},
        "latency_ms": latency_summary(latencies),
        "per_query": per_query,
    }


def print_summary(name: str, result: dict) -> None:
    keys = ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@10", "hit@5")
    print(f"\n== {name}  latency_ms={result['latency_ms']}")
    print(f"{'group':14s} {'n':>3s} " + " ".join(f"{key:>9s}" for key in keys))
    rows = [("overall", result["answerable_queries"], result["overall"]), ("cross_lingual", None, result["cross_lingual"])]
    rows += [(bucket, values["n"], values) for bucket, values in result["by_bucket"].items()]
    for label, count, values in rows:
        if values:
            print(f"{label:14s} {str(count or ''):>3s} " + " ".join(f"{values[key]:9.3f}" for key in keys))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality per bucket and language.")
    parser.add_argument("--queries", type=Path, default=REPO_ROOT / "data" / "smoke_queries.jsonl")
    parser.add_argument("--retrievers", nargs="+", choices=sorted(RETRIEVERS), default=["bm25", "dense"])
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--label", default="smoke")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    config = load_serving_config()
    fact_base = FactBase.load(config.paths.facts_dir)
    queries = load_queries(args.queries)
    unknown = sorted({fact_id for query in queries for fact_id in query.relevant_fact_ids} - set(fact_base.facts))
    if unknown:
        print(f"unknown fact ids in queries: {unknown}", file=sys.stderr)
        return 1
    chunks = read_chunks(config.paths.corpus_processed / CHUNKS_FILE_NAME)
    labels = chunk_fact_labels(chunks, config.paths.corpus_raw / MANIFEST_NAME)

    report = {
        "label": args.label,
        "queries_file": str(args.queries.relative_to(REPO_ROOT) if args.queries.is_relative_to(REPO_ROOT) else args.queries),
        "queries": len(queries),
        "chunks": len(chunks),
        "embedding_model": config.embedding.model,
        "top_k": args.top_k,
        "retrievers": {},
    }
    for name in args.retrievers:
        result = evaluate(RETRIEVERS[name](config), queries, labels, args.top_k)
        report["retrievers"][name] = result
        print_summary(name, result)

    output = args.output or REPO_ROOT / "results" / f"retrieval_{args.label}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
