from __future__ import annotations

from time import perf_counter

from rag.config import ServingConfig
from rag.langdetect import detect_language
from rag.normalize import normalize_for_dense
from rag.retrieval.bm25 import BM25Index
from rag.schemas import QuickSearchResponse, QuickSearchResult


def _elapsed_ms(start: float) -> float:
    return round((perf_counter() - start) * 1000, 2)


class QuickSearch:
    def __init__(self, config: ServingConfig) -> None:
        self.settings = config.quick_search
        self.index = BM25Index(config.opensearch)

    def search(self, query: str, top_k: int | None = None) -> QuickSearchResponse:
        started = perf_counter()
        language = detect_language(query)
        normalized = normalize_for_dense(query)
        preprocessing_ms = _elapsed_ms(started)

        search_started = perf_counter()
        hits = self.index.search(normalized, top_k or self.settings.top_k, self.settings.snippet_chars)
        search_ms = _elapsed_ms(search_started)

        results = [
            QuickSearchResult(
                doc_id=hit.chunk.doc_id,
                chunk_id=hit.chunk.chunk_id,
                title=hit.chunk.title,
                section=hit.chunk.section,
                snippet=hit.snippet,
                score=round(hit.score, 4),
                source=hit.chunk.source,
                page=hit.chunk.page,
            )
            for hit in hits
        ]
        response = QuickSearchResponse(
            query=query,
            language_detected=language,
            results=results,
            latency_ms={"preprocessing": preprocessing_ms, "search": search_ms, "total": 0.0},
        )
        response.latency_ms["total"] = _elapsed_ms(started)
        return response
