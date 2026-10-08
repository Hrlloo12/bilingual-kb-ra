from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

from rag.config import ServingConfig
from rag.generation import NOT_FOUND, Generator
from rag.grounding import ground
from rag.langdetect import detect_language
from rag.reranker import Reranker
from rag.retrieval.dense import Embedder
from rag.retrieval.hybrid import Candidate, HybridRetriever
from rag.schemas import Citation, Language, RetrievedPassage, SmartSearchResponse


def _elapsed_ms(start: float) -> float:
    return round((perf_counter() - start) * 1000, 2)


SNIPPET_CHARS = 240


def snippet_of(text: str, limit: int = SNIPPET_CHARS) -> str:
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[:limit].rsplit(" ", 1)[0] + "…"


def citation_for(citation_id: int, candidate: Candidate) -> Citation:
    chunk = candidate.chunk
    return Citation(
        id=citation_id,
        doc_id=chunk.doc_id,
        chunk_id=chunk.chunk_id,
        title=chunk.title,
        section=chunk.section,
        snippet=snippet_of(chunk.text),
        page=chunk.page,
        source=chunk.source,
        language=chunk.language,
        relevance=round(candidate.rerank_score, 4) if candidate.rerank_score is not None else None,
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

    def search(
        self, query: str, language: Language | None = None, on_ranked: Callable[[list[Candidate]], None] | None = None
    ) -> SmartSearchResponse:
        started = perf_counter()
        language = language or detect_language(query)
        latency = {"preprocessing": _elapsed_ms(started)}
        retrieval = self.retriever.retrieve(query)
        latency.update(retrieval.latency_ms)
        latency["retrieval"] = round(sum(retrieval.latency_ms.values()), 2)

        rerank_started = perf_counter()
        ranked = self.reranker.rerank(query, retrieval.candidates)
        latency["rerank"] = _elapsed_ms(rerank_started)
        if on_ranked is not None:
            on_ranked(ranked)

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
            generation_started = perf_counter()
            generation = self.generator.generate(query, language, [candidate.chunk for candidate in context])
            if generation.first_token_ms is not None:
                latency["generation_first_token"] = generation.first_token_ms
                latency["time_to_first_token"] = round((generation_started - started) * 1000 + generation.first_token_ms, 2)
            latency["generation_total"] = generation.latency_ms
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
