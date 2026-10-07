from __future__ import annotations

from time import perf_counter

from rag.config import ServingConfig
from rag.generation import NOT_FOUND, Generator
from rag.grounding import ground
from rag.langdetect import detect_language
from rag.reranker import Reranker
from rag.retrieval.dense import Embedder
from rag.retrieval.hybrid import Candidate, HybridRetriever
from rag.schemas import Citation, RetrievedPassage, SmartSearchResponse


def _elapsed_ms(start: float) -> float:
    return round((perf_counter() - start) * 1000, 2)


def citation_for(marker: int, candidate: Candidate) -> Citation:
    chunk = candidate.chunk
    return Citation(
        marker=marker,
        doc_id=chunk.doc_id,
        chunk_id=chunk.chunk_id,
        title=chunk.title,
        section=chunk.section,
        page=chunk.page,
        source=chunk.source,
        language=chunk.language,
    )


class SmartSearch:
    def __init__(
        self,
        config: ServingConfig,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        generator: Generator | None = None,
    ) -> None:
        self.settings = config.smart_search
        self.retriever = HybridRetriever(config, embedder)
        self.reranker = reranker or Reranker(config.reranker)
        self.generator = generator or Generator(config.generation)

    def select_context(self, ranked: list[Candidate]) -> list[Candidate]:
        settings = self.settings
        if not ranked or ranked[0].rerank_score < settings.abstain_threshold:
            return []
        return [candidate for candidate in ranked[: settings.context_top_k] if candidate.rerank_score >= settings.context_min_score] or ranked[:1]

    def search(self, query: str) -> SmartSearchResponse:
        started = perf_counter()
        language = detect_language(query)
        retrieval = self.retriever.retrieve(query)
        latency = dict(retrieval.latency_ms)

        rerank_started = perf_counter()
        ranked = self.reranker.rerank(query, retrieval.candidates)
        latency["rerank"] = _elapsed_ms(rerank_started)

        context = self.select_context(ranked)
        context_ids = {candidate.chunk.chunk_id for candidate in context}
        retrieved = [
            RetrievedPassage(
                chunk_id=candidate.chunk.chunk_id,
                doc_id=candidate.chunk.doc_id,
                title=candidate.chunk.title,
                section=candidate.chunk.section,
                rerank_score=round(candidate.rerank_score, 6),
                fused_score=round(candidate.fused_score, 6),
                bm25_rank=candidate.bm25_rank,
                dense_rank=candidate.dense_rank,
                in_context=candidate.chunk.chunk_id in context_ids,
            )
            for candidate in ranked[:10]
        ]

        status, answer, citations, reason, checks, usage = "not_found", NOT_FOUND, [], None, {}, {}
        if not context:
            reason = "low_retrieval_confidence"
        else:
            generation = self.generator.generate(query, language, [candidate.chunk for candidate in context])
            latency["generation"] = generation.latency_ms
            usage = {"prompt_tokens": generation.prompt_tokens, "completion_tokens": generation.completion_tokens}
            check_started = perf_counter()
            grounded = ground(generation.text, query, language, [candidate.chunk for candidate in context])
            checks = grounded.checks
            if grounded.not_found:
                reason = "generator_not_found"
            elif self.settings.abstain_on_unsupported_numbers and (checks["unsupported_numbers"] or checks["unsupported_codes"]):
                reason = "unsupported_numbers_or_codes"
                checks["rejected_answer"] = grounded.text
            else:
                status, answer = "answered", grounded.text
                markers = grounded.cited or [1]
                checks["citation_fallback"] = not grounded.cited
                citations = [citation_for(marker, context[marker - 1]) for marker in markers]
            latency["post_checks"] = _elapsed_ms(check_started)

        latency["total"] = _elapsed_ms(started)
        return SmartSearchResponse(
            query=query,
            language_detected=language,
            status=status,
            answer=answer,
            citations=citations,
            abstain_reason=reason,
            post_checks=checks,
            retrieved=retrieved,
            usage=usage,
            latency_ms=latency,
        )
