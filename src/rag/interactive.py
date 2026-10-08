from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from time import perf_counter

from rag.followups import FollowupSuggester
from rag.langdetect import detect_language
from rag.retrieval.hybrid import Candidate
from rag.rewrite import QueryRewriter
from rag.schemas import InteractiveResponse, Language, RewriteInfo
from rag.sessions import SessionStore, Turn, new_session_id
from rag.smart_search import SmartSearch

SMART_STAGES = (
    "preprocessing",
    "bm25",
    "embedding",
    "dense",
    "fusion",
    "retrieval",
    "rerank",
    "generation_first_token",
    "generation_total",
    "post_checks",
)


def _elapsed_ms(start: float) -> float:
    return round((perf_counter() - start) * 1000, 2)


class InteractiveSearch:
    def __init__(
        self,
        smart: SmartSearch,
        store: SessionStore,
        rewriter: QueryRewriter,
        suggester: FollowupSuggester | None = None,
        workers: int = 8,
    ) -> None:
        self.smart = smart
        self.store = store
        self.rewriter = rewriter
        self.suggester = suggester
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="followups") if suggester else None

    def _start_followups(self, query: str, language: Language, slot: dict) -> Callable[[list[Candidate]], None]:
        def on_ranked(ranked: list[Candidate]) -> None:
            chunks = [candidate.chunk for candidate in ranked]

            def run() -> list[str]:
                started = perf_counter()
                try:
                    return self.suggester.suggest(query, language, chunks)
                finally:
                    slot["ms"] = _elapsed_ms(started)

            slot["future"] = self.pool.submit(run)

        return on_ranked

    def search(self, query: str, session_id: str | None = None) -> InteractiveResponse:
        started = perf_counter()
        session_id = session_id or new_session_id()
        history = self.store.history(session_id)
        latency = {"memory_read": _elapsed_ms(started)}
        language = detect_language(query)
        rewrite = self.rewriter.rewrite(query, history)
        latency["query_rewrite"] = rewrite.latency_ms

        slot: dict = {}
        on_ranked = self._start_followups(query, language, slot) if self.pool else None
        smart_started = perf_counter()
        result = self.smart.search(rewrite.query, language=language, on_ranked=on_ranked)
        for stage in SMART_STAGES:
            if stage in result.latency_ms:
                latency[stage] = result.latency_ms[stage]
        if "time_to_first_token" in result.latency_ms:
            latency["time_to_first_token"] = round((smart_started - started) * 1000 + result.latency_ms["time_to_first_token"], 2)
        latency["smart_search"] = result.latency_ms["total"]

        followups: list[str] = []
        future: Future | None = slot.get("future")
        if future is not None:
            wait_started = perf_counter()
            followups = future.result()
            latency["followups"] = slot.get("ms", 0.0)
            latency["followups_wait"] = _elapsed_ms(wait_started)

        memory_started = perf_counter()
        self.store.append(
            session_id,
            Turn(
                query=query,
                standalone_query=rewrite.query,
                language=language,
                status=result.status,
                answer=result.answer,
                cited_chunk_ids=[citation.chunk_id for citation in result.citations],
            ),
        )
        latency["memory_write"] = _elapsed_ms(memory_started)
        latency["total"] = _elapsed_ms(started)
        return InteractiveResponse(
            session_id=session_id,
            turn=len(history) + 1,
            new_session=not history,
            query=query,
            rewritten_query=rewrite.query,
            rewrite=RewriteInfo(applied=rewrite.applied, reason=rewrite.reason, latency_ms=rewrite.latency_ms, fallback=rewrite.fallback),
            language_detected=language,
            status=result.status,
            answer=result.answer,
            citations=result.citations,
            suggested_followups=followups,
            abstain_reason=result.abstain_reason,
            post_checks=result.post_checks,
            retrieved=result.retrieved,
            usage=result.usage,
            latency_ms=latency,
        )
