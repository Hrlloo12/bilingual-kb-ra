from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from rag.config import REPO_ROOT, load_serving_config
from rag.evaluation.metrics import latency_summary
from rag.ingestion import CHUNKS_FILE_NAME, read_chunks
from rag.scaling import make_distractors

UPSERT_BATCH = 1000


def gpu_memory_mib() -> int | None:
    import subprocess

    try:
        output = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
        return int(output.stdout.split()[0])
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError, ValueError):
        return None


def wait_until_settled(dense, timeout_s: float = 1800, stable_checks: int = 5) -> tuple[float, dict]:
    started = time.perf_counter()
    stable, previous = 0, None
    while time.perf_counter() - started < timeout_s:
        info = dense.client.get_collection(dense.collection)
        state = (str(info.status).lower(), str(info.optimizer_status).lower(), info.indexed_vectors_count or 0, info.segments_count)
        settled = state[0].endswith("green") and state[1].endswith("ok")
        stable = stable + 1 if settled and state == previous else 0
        previous = state
        if stable >= stable_checks:
            points = info.points_count or 0
            indexed = info.indexed_vectors_count or 0
            mode = "hnsw" if indexed >= points * 0.99 else "exact" if indexed == 0 else "partial hnsw"
            return time.perf_counter() - started, {"points": points, "indexed_vectors": indexed, "segments": info.segments_count, "search": mode}
        time.sleep(2)
    raise TimeoutError(f"{dense.collection} not settled after {timeout_s}s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure Smart AI Search latency as the corpus grows with synthetic distractor chunks.")
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 10000, 100000])
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    from rag.reranker import Reranker
    from rag.retrieval.bm25 import BM25Index
    from rag.retrieval.dense import DenseIndex, Embedder
    from rag.smart_search import SmartSearch

    config = load_serving_config()
    real = read_chunks(config.paths.corpus_processed / CHUNKS_FILE_NAME)
    largest = max(args.sizes)
    distractors = make_distractors(real, largest - len(real))
    embedder = Embedder(config.embedding.model_copy(update={"batch_size": 64}))
    reranker = Reranker(config.reranker)

    started = time.perf_counter()
    vectors = embedder.encode_passages(real + distractors)
    embedding_seconds = time.perf_counter() - started
    rows = [json.loads(line) for line in (REPO_ROOT / "data" / "test.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    queries = [row["query"] for row in rows[: args.queries]]

    report = {
        "note": "synthetic distractors are used for latency only; retrieval quality is never measured on these indexes",
        "real_chunks": len(real),
        "embedding": {"chunks": len(real) + len(distractors), "seconds": round(embedding_seconds, 1), "chunks_per_second": round((len(real) + len(distractors)) / embedding_seconds, 1)},
        "queries": len(queries),
        "sizes": [],
    }
    for size in args.sizes:
        corpus = real + distractors[: size - len(real)]
        name = f"kb_scale_{size}"
        scaled = config.model_copy(
            update={"opensearch": config.opensearch.model_copy(update={"index": name, "timeout_s": 120}), "qdrant": config.qdrant.model_copy(update={"collection": name, "timeout_s": 120})}
        )
        bm25, dense = BM25Index(scaled.opensearch), DenseIndex(scaled.qdrant)
        started = time.perf_counter()
        bm25.recreate()
        bm25.add(corpus)
        bm25_seconds = time.perf_counter() - started
        started = time.perf_counter()
        dense.recreate(embedder.dimension)
        for start in range(0, len(corpus), UPSERT_BATCH):
            dense.add(corpus[start : start + UPSERT_BATCH], vectors[start : start + UPSERT_BATCH])
        upsert_seconds = time.perf_counter() - started
        settle_seconds, qdrant_state = wait_until_settled(dense)

        search = SmartSearch(scaled, embedder=embedder, reranker=reranker)
        for warmup in queries[:3]:
            search.search(warmup)
        stages: dict[str, list[float]] = {}
        statuses: dict[str, int] = {}
        for query in queries:
            response = search.search(query)
            statuses[response.status] = statuses.get(response.status, 0) + 1
            for stage, value in response.latency_ms.items():
                stages.setdefault(stage, []).append(value)
        entry = {
            "chunks": size,
            "bm25_docs": bm25.count(),
            "qdrant_points": dense.count(),
            "index_seconds": {"bm25_bulk": round(bm25_seconds, 1), "qdrant_upsert": round(upsert_seconds, 1), "qdrant_settle_wait": round(settle_seconds, 1)},
            "qdrant": qdrant_state,
            "opensearch_store": bm25.client.cat.indices(index=name, h="store.size", format="json")[0]["store.size"],
            "latency_ms": {stage: latency_summary(values) for stage, values in stages.items()},
            "statuses": statuses,
            "gpu_memory_mib": gpu_memory_mib(),
        }
        report["sizes"].append(entry)
        print(json.dumps({"chunks": size, "total": entry["latency_ms"]["total"], "retrieval_p50": {stage: entry["latency_ms"][stage]["p50"] for stage in ("bm25", "embedding", "dense", "rerank")}}))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        bm25.client.indices.delete(index=name)
        dense.client.delete_collection(name)
    del vectors
    return 0


if __name__ == "__main__":
    sys.exit(main())
