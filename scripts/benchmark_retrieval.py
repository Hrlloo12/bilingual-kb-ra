from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

from rag.config import REPO_ROOT, ServingConfig, load_serving_config
from rag.evaluation.metrics import latency_summary
from rag.evaluation.relevance import RetrievalQuery, chunk_fact_labels, load_queries
from rag.evaluation.report import evaluate, print_summary
from rag.facts import FactBase
from rag.ingestion import CHUNKS_FILE_NAME, MANIFEST_NAME, read_chunks
from rag.normalize import normalize_for_dense
from rag.retrieval.fusion import weighted_rrf

RETRIEVER_NAMES = ("bm25", "dense", "hybrid", "hybrid_rerank")


class RankingBuilder:
    def __init__(self, config: ServingConfig, needs_dense: bool) -> None:
        from rag.retrieval.bm25 import BM25Index

        self.config = config
        self.bm25 = BM25Index(config.opensearch)
        self.embedder = None
        self.dense = None
        if needs_dense:
            from rag.retrieval.dense import DenseIndex, Embedder

            self.embedder = Embedder(config.embedding)
            self.dense = DenseIndex(config.qdrant)

    def bm25_rankings(self, queries: list[RetrievalQuery], top_k: int) -> tuple[list[list[str]], list[float]]:
        rankings, latencies = [], []
        self.bm25.search(normalize_for_dense(queries[0].query), top_k)
        for query in queries:
            started = perf_counter()
            hits = self.bm25.search(normalize_for_dense(query.query), top_k)
            latencies.append((perf_counter() - started) * 1000)
            rankings.append([hit.chunk.chunk_id for hit in hits])
        return rankings, latencies

    def dense_rankings(self, queries: list[RetrievalQuery], top_k: int) -> list[list[str]]:
        vectors = self.embedder.encode_queries([query.query for query in queries])
        return [[hit.chunk.chunk_id for hit in self.dense.search(vector, top_k)] for vector in vectors]

    def hybrid_rankings(self, bm25: list[list[str]], dense: list[list[str]], top_k: int) -> list[list[str]]:
        settings = self.config.retrieval
        weights = {"bm25": settings.bm25_weight, "dense": settings.dense_weight}
        return [
            [chunk_id for chunk_id, _ in weighted_rrf({"bm25": lexical, "dense": semantic}, weights, settings.rrf_k, top_k)]
            for lexical, semantic in zip(bm25, dense, strict=True)
        ]

    def reranked(self, queries: list[RetrievalQuery], candidates: list[list[str]], chunks: dict) -> list[list[str]]:
        from rag.reranker import Reranker
        from rag.retrieval.dense import passage_text

        reranker = Reranker(self.config.reranker)
        rankings = []
        for query, ids in zip(queries, candidates, strict=True):
            scores = reranker.score(query.query, [passage_text(chunks[chunk_id]) for chunk_id in ids])
            rankings.append([chunk_id for _, chunk_id in sorted(zip(scores, ids, strict=True), key=lambda pair: -pair[0])])
        return rankings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality per bucket and language.")
    parser.add_argument("--queries", type=Path, default=REPO_ROOT / "data" / "test.jsonl")
    parser.add_argument("--retrievers", nargs="+", choices=RETRIEVER_NAMES, default=list(RETRIEVER_NAMES))
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--label", default="test")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--stage", choices=("before", "after"), default=None, help="Also record results in the before/after fine-tuning metrics file.")
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

    needs_dense = bool({"dense", "hybrid", "hybrid_rerank"} & set(args.retrievers))
    builder = RankingBuilder(config, needs_dense)
    retrieval = config.retrieval
    bm25, bm25_latency = builder.bm25_rankings(queries, max(args.top_k, retrieval.bm25_top_k))
    dense = builder.dense_rankings(queries, max(args.top_k, retrieval.dense_top_k)) if needs_dense else []

    report = {
        "label": args.label,
        "queries_file": str(args.queries.relative_to(REPO_ROOT) if args.queries.is_relative_to(REPO_ROOT) else args.queries),
        "queries": len(queries),
        "unanswerable_excluded": sum(1 for query in queries if not query.answerable),
        "chunks": len(chunks),
        "embedding_model": config.embedding.model if needs_dense else None,
        "qdrant_collection": config.qdrant.collection if needs_dense else None,
        "fusion": retrieval.model_dump(),
        "top_k": args.top_k,
        "retrievers": {},
    }
    for name in args.retrievers:
        if name == "bm25":
            result = evaluate([ranked[: args.top_k] for ranked in bm25], queries, labels)
            result["latency_ms"] = latency_summary(bm25_latency)
        elif name == "dense":
            result = evaluate([ranked[: args.top_k] for ranked in dense], queries, labels)
        elif name == "hybrid":
            result = evaluate(builder.hybrid_rankings(bm25, dense, args.top_k), queries, labels)
        else:
            pool = builder.hybrid_rankings(bm25, dense, retrieval.candidates)
            result = evaluate(builder.reranked(queries, pool, {chunk.chunk_id: chunk for chunk in chunks}), queries, labels)
            result["reranker_model"] = config.reranker.model
            result["candidates"] = retrieval.candidates
        report["retrievers"][name] = result
        print_summary(name, result)

    output = args.output or REPO_ROOT / "results" / f"retrieval_{args.label}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved {output}")
    if args.stage:
        update_stage_metrics(args.stage, report)
    return 0


def update_stage_metrics(stage: str, report: dict) -> None:
    path = REPO_ROOT / "results" / f"{stage}_finetuning_metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"stage": stage}
    metrics["retrieval"] = {
        "queries_file": report["queries_file"],
        "embedding_model": report["embedding_model"],
        "answerable_queries": next(iter(report["retrievers"].values()))["answerable_queries"],
        "retrievers": {
            name: {key: result[key] for key in ("overall", "cross_lingual", "by_language", "by_bucket")}
            | ({"latency_ms": result["latency_ms"]} if "latency_ms" in result else {})
            for name, result in report["retrievers"].items()
        },
    }
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"updated {path}")


if __name__ == "__main__":
    sys.exit(main())
