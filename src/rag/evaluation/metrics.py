from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

RECALL_CUTOFFS = (1, 5, 10, 20)


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant)


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return float(any(chunk_id in relevant for chunk_id in ranked[:k]))


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    for rank, chunk_id in enumerate(ranked, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(rank + 1) for rank, chunk_id in enumerate(ranked[:k], start=1) if chunk_id in relevant)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def query_metrics(ranked: Sequence[str], relevant: set[str]) -> dict[str, float]:
    metrics = {f"recall@{k}": recall_at_k(ranked, relevant, k) for k in RECALL_CUTOFFS}
    metrics.update({f"hit@{k}": hit_at_k(ranked, relevant, k) for k in RECALL_CUTOFFS})
    metrics["mrr"] = reciprocal_rank(ranked, relevant)
    metrics["ndcg@10"] = ndcg_at_k(ranked, relevant, 10)
    return metrics


def aggregate(per_query: list[dict[str, float]]) -> dict[str, float]:
    if not per_query:
        return {}
    return {key: round(float(np.mean([metrics[key] for metrics in per_query])), 4) for key in per_query[0]}


def latency_summary(samples_ms: list[float]) -> dict[str, float]:
    values = np.asarray(samples_ms)
    return {
        "avg": round(float(values.mean()), 2),
        "p50": round(float(np.percentile(values, 50)), 2),
        "p95": round(float(np.percentile(values, 95)), 2),
        "max": round(float(values.max()), 2),
    }
