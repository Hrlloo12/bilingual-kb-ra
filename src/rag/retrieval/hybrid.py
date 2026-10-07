from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from rag.config import ServingConfig
from rag.normalize import normalize_for_dense
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.dense import DenseIndex, Embedder
from rag.retrieval.fusion import weighted_rrf
from rag.schemas import Chunk


@dataclass
class Candidate:
    chunk: Chunk
    fused_score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    rerank_score: float | None = None


@dataclass
class HybridResult:
    candidates: list[Candidate]
    latency_ms: dict[str, float] = field(default_factory=dict)


def _elapsed_ms(start: float) -> float:
    return round((perf_counter() - start) * 1000, 2)


class HybridRetriever:
    def __init__(self, config: ServingConfig, embedder: Embedder | None = None) -> None:
        self.settings = config.retrieval
        self.bm25 = BM25Index(config.opensearch)
        self.embedder = embedder or Embedder(config.embedding)
        self.dense = DenseIndex(config.qdrant)

    def retrieve(self, query: str, limit: int | None = None) -> HybridResult:
        settings = self.settings
        latency: dict[str, float] = {}
        normalized = normalize_for_dense(query)

        started = perf_counter()
        lexical_depth = max(settings.bm25_top_k if settings.bm25_weight > 0 else 0, settings.bm25_extra_candidates)
        lexical = self.bm25.search(normalized, lexical_depth) if lexical_depth else []
        latency["bm25"] = _elapsed_ms(started)

        started = perf_counter()
        vector = self.embedder.encode_queries([query])[0]
        latency["embedding"] = _elapsed_ms(started)

        started = perf_counter()
        semantic = self.dense.search(vector, settings.dense_top_k)
        latency["dense"] = _elapsed_ms(started)

        started = perf_counter()
        chunks = {hit.chunk.chunk_id: hit.chunk for hit in [*lexical, *semantic]}
        bm25_ranks = {hit.chunk.chunk_id: rank for rank, hit in enumerate(lexical, start=1)}
        dense_hits = {hit.chunk.chunk_id: (rank, hit.score) for rank, hit in enumerate(semantic, start=1)}
        weights = {"bm25": settings.bm25_weight, "dense": settings.dense_weight}
        sources = {"bm25": [hit.chunk.chunk_id for hit in lexical[: settings.bm25_top_k]], "dense": [hit.chunk.chunk_id for hit in semantic]}
        fused = weighted_rrf(
            {name: ranked for name, ranked in sources.items() if weights[name] > 0}, weights, settings.rrf_k, limit or settings.candidates
        )
        fused_ids = {chunk_id for chunk_id, _ in fused}
        fused += [(hit.chunk.chunk_id, 0.0) for hit in lexical[: settings.bm25_extra_candidates] if hit.chunk.chunk_id not in fused_ids]
        candidates = [
            Candidate(
                chunk=chunks[chunk_id],
                fused_score=score,
                bm25_rank=bm25_ranks.get(chunk_id),
                dense_rank=dense_hits.get(chunk_id, (None, None))[0],
                dense_score=dense_hits.get(chunk_id, (None, None))[1],
            )
            for chunk_id, score in fused
        ]
        latency["fusion"] = _elapsed_ms(started)
        return HybridResult(candidates=candidates, latency_ms=latency)
