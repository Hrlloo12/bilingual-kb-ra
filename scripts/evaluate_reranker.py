from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from rag.config import REPO_ROOT, load_serving_config
from rag.evaluation.relevance import chunk_fact_labels, load_queries, relevant_chunk_ids
from rag.evaluation.report import evaluate, print_summary
from rag.ingestion import CHUNKS_FILE_NAME, MANIFEST_NAME, read_chunks
from rag.retrieval.fusion import weighted_rrf

SELECTION_METRIC = "ndcg@10"
POOLS = ("dense_ft", "hybrid_equal", "dense_ft+bm25_top10")


def load_scores(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {query_id: dict((chunk_id, score) for chunk_id, score in row["scores"]) for query_id, row in payload.items()}


def hybrid_pool(rankings: dict[str, list], bm25_weight: float, rrf_k: int, depth: int, size: int) -> list[str]:
    sources = {"bm25": [chunk_id for chunk_id, _ in rankings["bm25"][:depth]], "dense_ft": [chunk_id for chunk_id, _ in rankings["dense_ft"][:depth]]}
    weights = {"bm25": bm25_weight, "dense_ft": 1.0}
    return [chunk_id for chunk_id, _ in weighted_rrf({name: ranked for name, ranked in sources.items() if weights[name] > 0}, weights, rrf_k, size)]


def system_name(pool: str, scorer: str | None) -> str:
    return pool if scorer is None else f"{pool}+rerank_{scorer}"


def pool_recall(pool: dict[str, list[str]], queries, labels) -> float:
    values = [float(any(chunk_id in relevant_chunk_ids(query, labels) for chunk_id in pool[query.id])) for query in queries if query.answerable]
    return round(sum(values) / len(values), 4)


def rerank(pool: list[str], scores: dict[str, float]) -> list[str]:
    return sorted(pool, key=lambda chunk_id: -scores[chunk_id])


def calibrate(top_scores: list[float], answerable: list[bool]) -> dict:
    scores = np.asarray(top_scores)
    labels = np.asarray(answerable)
    best = None
    for threshold in sorted({0.0, *np.round(scores, 6).tolist()}):
        kept = scores >= threshold
        answer_recall = float(kept[labels].mean())
        abstain_recall = float((~kept[~labels]).mean())
        balanced = (answer_recall + abstain_recall) / 2
        row = {"threshold": threshold, "balanced_accuracy": round(balanced, 4), "answerable_kept": round(answer_recall, 4), "unanswerable_rejected": round(abstain_recall, 4)}
        if best is None or balanced > best["balanced_accuracy"]:
            best = row
    return best


def apply_threshold(top_scores: list[float], answerable: list[bool], threshold: float) -> dict:
    scores = np.asarray(top_scores)
    labels = np.asarray(answerable)
    kept = scores >= threshold
    answer_recall = float(kept[labels].mean())
    abstain_recall = float((~kept[~labels]).mean())
    return {
        "threshold": threshold,
        "balanced_accuracy": round((answer_recall + abstain_recall) / 2, 4),
        "answerable_kept": round(answer_recall, 4),
        "unanswerable_rejected": round(abstain_recall, 4),
        "answerable_n": int(labels.sum()),
        "unanswerable_n": int((~labels).sum()),
    }


def score_distribution(top_scores: list[float], answerable: list[bool]) -> dict:
    scores = np.asarray(top_scores)
    labels = np.asarray(answerable)
    summary = {}
    for name, mask in (("answerable", labels), ("unanswerable", ~labels)):
        values = scores[mask]
        summary[name] = {key: round(float(np.percentile(values, q)), 4) for key, q in (("p05", 5), ("p25", 25), ("p50", 50), ("p75", 75), ("p95", 95))}
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate base and fine-tuned reranker on several candidate pools and calibrate abstention.")
    parser.add_argument("--rankings-dir", type=Path, default=REPO_ROOT / "results" / "rankings")
    parser.add_argument("--scores-dir", type=Path, default=REPO_ROOT / "results" / "reranker")
    parser.add_argument("--fusion", type=Path, default=REPO_ROOT / "results" / "fusion" / "fusion_grid_validation.json")
    args = parser.parse_args(argv)

    config = load_serving_config()
    pool_size = config.retrieval.candidates
    fusion = json.loads(args.fusion.read_text(encoding="utf-8"))
    selected = fusion["selected"]
    depth = fusion["depth"]
    chunks = read_chunks(config.paths.corpus_processed / CHUNKS_FILE_NAME)
    labels = chunk_fact_labels(chunks, config.paths.corpus_raw / MANIFEST_NAME)
    splits = {split: load_queries(REPO_ROOT / "data" / f"{split}.jsonl") for split in ("validation", "test")}
    rankings = {split: json.loads((args.rankings_dir / f"{split}.json").read_text(encoding="utf-8"))["queries"] for split in splits}
    scorers = {name: load_scores(args.scores_dir / f"scores_{name}.json") for name in ("base", "finetuned")}

    pools = {
        split: {
            "dense_ft": {query.id: [chunk_id for chunk_id, _ in rankings[split][query.id]["dense_ft"][:pool_size]] for query in queries},
            "hybrid_equal": {query.id: hybrid_pool(rankings[split][query.id], 1.0, 60, depth, pool_size) for query in queries},
            "dense_ft+bm25_top10": {
                query.id: list(dict.fromkeys([chunk_id for chunk_id, _ in rankings[split][query.id]["dense_ft"][:pool_size]] + [chunk_id for chunk_id, _ in rankings[split][query.id]["bm25"][:10]]))
                for query in queries
            },
        }
        for split, queries in splits.items()
    }

    reports = {split: {"split": split, "pool_size": pool_size, "fusion": selected, "systems": {}} for split in splits}
    for split, queries in splits.items():
        report = reports[split]
        for pool_name in POOLS:
            result = evaluate([pools[split][pool_name][query.id][:20] for query in queries], queries, labels)
            result["pool_recall"] = pool_recall(pools[split][pool_name], queries, labels)
            report["systems"][pool_name] = result
            print_summary(f"{split} / {pool_name}", result)
            for scorer_name, scores in scorers.items():
                ranked = [rerank(pools[split][pool_name][query.id], scores[query.id])[:20] for query in queries]
                name = f"{pool_name}+rerank_{scorer_name}"
                result = evaluate(ranked, queries, labels)
                report["systems"][name] = result
                print_summary(f"{split} / {name}", result)

    validation_systems = reports["validation"]["systems"]
    options = [(pool, scorer) for pool in POOLS for scorer in (None, "base", "finetuned")]
    pool_choice, chosen = max(
        options, key=lambda option: (validation_systems[system_name(*option)]["overall"][SELECTION_METRIC], -options.index(option))
    )
    print(f"\nselected on validation {SELECTION_METRIC}: pool={pool_choice} reranker={chosen}")

    abstention = {"selected_pool": pool_choice, "selected_reranker": chosen, "criterion": "max balanced accuracy on validation (answerable kept vs unanswerable rejected), top-1 reranker score"}
    for scorer_name, scores in scorers.items():
        entry = {}
        threshold = None
        for split in ("validation", "test"):
            queries = splits[split]
            top = [max(scores[query.id][chunk_id] for chunk_id in pools[split][pool_choice][query.id]) for query in queries]
            answerable = [query.answerable for query in queries]
            if split == "validation":
                threshold = calibrate(top, answerable)["threshold"]
            entry[split] = {"at_validation_threshold": apply_threshold(top, answerable, threshold), "top_score_distribution": score_distribution(top, answerable)}
        abstention[scorer_name] = entry
        print(f"abstention {scorer_name}: {json.dumps({split: entry[split]['at_validation_threshold'] for split in entry})}")

    args.scores_dir.mkdir(parents=True, exist_ok=True)
    for split, report in reports.items():
        report["selected"] = {"pool": pool_choice, "reranker": chosen, "metric": SELECTION_METRIC}
        (args.scores_dir / f"reranker_comparison_{split}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.scores_dir / "abstention_calibration.json").write_text(json.dumps(abstention, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
