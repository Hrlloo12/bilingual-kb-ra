from __future__ import annotations

from collections import defaultdict


def weighted_rrf(
    rankings: dict[str, list[str]], weights: dict[str, float], k: int = 60, limit: int | None = None
) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for name, ranked in rankings.items():
        weight = weights.get(name, 1.0)
        for rank, item in enumerate(ranked, start=1):
            scores[item] += weight / (k + rank)
    fused = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    return fused[:limit] if limit else fused
