from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag.config import REPO_ROOT, load_serving_config
from rag.evaluation.relevance import load_queries
from rag.normalize import normalize_for_dense
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.dense import DenseIndex, Embedder

DENSE_VARIANTS = {
    "dense_base": ("Qwen/Qwen3-Embedding-0.6B", "kb_chunks"),
    "dense_ft": (None, None),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export BM25 and dense rankings with scores for offline fusion and reranking.")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "rankings")
    args = parser.parse_args(argv)

    config = load_serving_config()
    queries = {split: load_queries(REPO_ROOT / "data" / f"{split}.jsonl") for split in args.splits}
    rankings: dict[str, dict[str, dict[str, list]]] = {split: {query.id: {} for query in rows} for split, rows in queries.items()}

    bm25 = BM25Index(config.opensearch)
    for split, rows in queries.items():
        for query in rows:
            hits = bm25.search(normalize_for_dense(query.query), args.top_k)
            rankings[split][query.id]["bm25"] = [[hit.chunk.chunk_id, round(hit.score, 4)] for hit in hits]

    for name, (model, collection) in DENSE_VARIANTS.items():
        embedding = config.embedding.model_copy(update={"model": model or config.embedding.model})
        qdrant = config.qdrant.model_copy(update={"collection": collection or config.qdrant.collection})
        embedder = Embedder(embedding)
        index = DenseIndex(qdrant)
        for split, rows in queries.items():
            vectors = embedder.encode_queries([query.query for query in rows])
            for query, vector in zip(rows, vectors, strict=True):
                hits = index.search(vector, args.top_k)
                rankings[split][query.id][name] = [[hit.chunk.chunk_id, round(hit.score, 4)] for hit in hits]
        print(f"{name}: {embedding.model} / {qdrant.collection}")
        del embedder

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, by_query in rankings.items():
        payload = {
            "split": split,
            "top_k": args.top_k,
            "retrievers": {"bm25": config.opensearch.index, **{name: (model or config.embedding.model) for name, (model, _) in DENSE_VARIANTS.items()}},
            "queries": by_query,
        }
        path = args.output_dir / f"{split}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        print(f"saved {path} ({len(by_query)} queries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
