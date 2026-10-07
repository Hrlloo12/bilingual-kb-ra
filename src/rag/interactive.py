from __future__ import annotations

from time import perf_counter

from rag.langdetect import detect_language
from rag.rewrite import QueryRewriter
from rag.schemas import InteractiveResponse, RewriteInfo
from rag.sessions import SessionStore, Turn, new_session_id
from rag.smart_search import SmartSearch


class InteractiveSearch:
    def __init__(self, smart: SmartSearch, store: SessionStore, rewriter: QueryRewriter) -> None:
        self.smart = smart
        self.store = store
        self.rewriter = rewriter

    def search(self, query: str, session_id: str | None = None) -> InteractiveResponse:
        started = perf_counter()
        session_id = session_id or new_session_id()
        history = self.store.history(session_id)
        memory_read_ms = round((perf_counter() - started) * 1000, 2)
        language = detect_language(query)
        rewrite = self.rewriter.rewrite(query, history)
        result = self.smart.search(rewrite.query, language=language)

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
        memory_ms = round((perf_counter() - memory_started) * 1000, 2)
        latency = {
            "memory_read": memory_read_ms,
            "rewrite": rewrite.latency_ms,
            "smart_search": result.latency_ms["total"],
            "memory_write": memory_ms,
            "total": round((perf_counter() - started) * 1000, 2),
        }
        return InteractiveResponse(
            session_id=session_id,
            turn=len(history) + 1,
            new_session=not history,
            query=query,
            rewrite=RewriteInfo(
                applied=rewrite.applied,
                reason=rewrite.reason,
                standalone_query=rewrite.query,
                latency_ms=rewrite.latency_ms,
                fallback=rewrite.fallback,
            ),
            result=result,
            latency_ms=latency,
        )
