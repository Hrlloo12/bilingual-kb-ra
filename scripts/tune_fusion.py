from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

from rag.config import REPO_ROOT, load_serving_config
from rag.evaluation.metrics import recall_at_k
from rag.evaluation.relevance import chunk_fact_labels, load_queries, relevant_chunk_ids
from rag.evaluation.report import evaluate, print_summary
from rag.ingestion import CHUNKS_FILE_NAME, MANIFEST_NAME, read_chunks
from rag.retrieval.fusion import weighted_rrf

BM25_WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
RRF_KS = (5, 10, 20, 30, 60, 100)
SELECTION_METRIC = "ndcg@10"


def load_rankings(path: Path) -> dict[str, dict[str, list[str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        query_id: {name: [chunk_id for chunk_id, _ in hits] for name, hits in retrievers.items()}
        for query_id, retrievers in payload["queries"].items()
    }


def fuse(rankings: dict[str, list[str]], bm25_weight: float, rrf_k: int, depth: int) -> list[str]:
    weights = {"bm25": bm25_weight, "dense_ft": 1.0}
    sources = {name: rankings[name][:depth] for name in weights if weights[name] > 0}
    return [chunk_id for chunk_id, _ in weighted_rrf(sources, weights, rrf_k)]


def candidate_recall(rankings: list[list[str]], queries, labels, cutoff: int) -> float:
    values = [recall_at_k(ranked, relevant_chunk_ids(query, labels), cutoff) for query, ranked in zip(queries, rankings) if query.answerable]
    return round(sum(values) / len(values), 4)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tune weighted RRF on validation and compare retrievers on each split.")
    parser.add_argument("--rankings-dir", type=Path, default=REPO_ROOT / "results" / "rankings")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "fusion")
    args = parser.parse_args(argv)

    config = load_serving_config()
    retrieval = config.retrieval
    depth = min(retrieval.bm25_top_k, retrieval.dense_top_k)
    chunks = read_chunks(config.paths.corpus_processed / CHUNKS_FILE_NAME)
    labels = chunk_fact_labels(chunks, config.paths.corpus_raw / MANIFEST_NAME)
    splits = {split: load_queries(REPO_ROOT / "data" / f"{split}.jsonl") for split in ("validation", "test")}
    rankings = {split: load_rankings(args.rankings_dir / f"{split}.json") for split in splits}

    validation = splits["validation"]
    grid = []
    for bm25_weight, rrf_k in itertools.product(BM25_WEIGHTS, RRF_KS):
        fused = [fuse(rankings["validation"][query.id], bm25_weight, rrf_k, depth) for query in validation]
        result = evaluate(fused, validation, labels)
        grid.append(
            {
                "bm25_weight": bm25_weight,
                "dense_weight": 1.0,
                "rrf_k": rrf_k,
                **{key: result["overall"][key] for key in ("ndcg@10", "mrr", "recall@1", "recall@5", "recall@20")},
                f"recall@{retrieval.candidates}": candidate_recall(fused, validation, labels, retrieval.candidates),
                "cross_lingual_mrr": result["cross_lingual"]["mrr"],
            }
        )
    candidate_key = f"recall@{retrieval.candidates}"
    selected = max(grid, key=lambda row: (row[SELECTION_METRIC], row[candidate_key], row["mrr"], -row["bm25_weight"], -abs(row["rrf_k"] - 60)))
    print(f"selected on validation {SELECTION_METRIC}: {selected}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "fusion_grid_validation.json").write_text(
        json.dumps({"selection_metric": SELECTION_METRIC, "depth": depth, "selected": selected, "grid": grid}, indent=2), encoding="utf-8"
    )

    systems = {
        "bm25": lambda ranked: ranked["bm25"],
        "dense_base": lambda ranked: ranked["dense_base"],
        "dense_ft": lambda ranked: ranked["dense_ft"],
        "hybrid_equal": lambda ranked: fuse(ranked, 1.0, 60, depth),
        "hybrid_tuned": lambda ranked: fuse(ranked, selected["bm25_weight"], selected["rrf_k"], depth),
    }
    for split, queries in splits.items():
        report = {"split": split, "fusion_selected_on": "validation", "selected": selected, "systems": {}}
        for name, build in systems.items():
            fused = [build(rankings[split][query.id]) for query in queries]
            result = evaluate([ranked[:20] for ranked in fused], queries, labels)
            result[candidate_key] = candidate_recall(fused, queries, labels, retrieval.candidates)
            report["systems"][name] = result
            print_summary(f"{split} / {name}", result)
        (args.output_dir / f"retrieval_comparison_{split}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
