from __future__ import annotations

import logging
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from opensearchpy.exceptions import ConnectionError as SearchConnectionError
from pydantic import BaseModel, Field, field_validator
from valkey.exceptions import ConnectionError as SessionConnectionError

from rag.config import REPO_ROOT, ServingConfig, load_serving_config
from rag.interactive import InteractiveSearch
from rag.quick_search import QuickSearch
from rag.schemas import InteractiveResponse, QuickSearchResponse, SmartSearchResponse
from rag.sessions import SESSION_ID_PATTERN, SessionStore
from rag.smart_search import SmartSearch

logger = logging.getLogger("rag.api")
WEB_DIR = REPO_ROOT / "demo" / "web"
MAX_QUERY_CHARS = 1000
ALL_MODES = ("quick_search", "smart_ai_search", "interactive")
MODE_ALIASES = {"smart_search": "smart_ai_search"}


class SearchRequest(BaseModel):
    mode: Literal["quick_search", "smart_ai_search", "smart_search", "interactive"]
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    session_id: str | None = Field(default=None, pattern=SESSION_ID_PATTERN.pattern)
    top_k: int | None = Field(default=None, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value.strip()

    @field_validator("mode")
    @classmethod
    def _canonical_mode(cls, value: str) -> str:
        return MODE_ALIASES.get(value, value)


@dataclass
class Components:
    quick: QuickSearch
    smart: SmartSearch | None = None
    interactive: InteractiveSearch | None = None
    store: SessionStore | None = None
    checks: dict[str, Callable[[], bool]] = field(default_factory=dict)


def enabled_modes() -> tuple[str, ...]:
    requested = [MODE_ALIASES.get(mode.strip(), mode.strip()) for mode in os.environ.get("API_MODES", ",".join(ALL_MODES)).split(",") if mode.strip()]
    unknown = sorted(set(requested) - set(ALL_MODES))
    if unknown or not requested:
        raise ValueError(f"API_MODES must list modes from {ALL_MODES}, got {requested}")
    return tuple(mode for mode in ALL_MODES if mode in requested)


def build_components(config: ServingConfig, modes: tuple[str, ...] = ALL_MODES) -> Components:
    quick = QuickSearch(config)
    checks: dict[str, Callable[[], bool]] = {"opensearch": lambda: quick.index.count() > 0}
    if modes == ("quick_search",):
        return Components(quick=quick, checks=checks)

    from rag.followups import FollowupSuggester
    from rag.reranker import Reranker
    from rag.retrieval.dense import Embedder
    from rag.rewrite import QueryRewriter

    smart = SmartSearch(config, embedder=Embedder(config.embedding), reranker=Reranker(config.reranker))
    store = SessionStore.from_url(config.valkey.url, config.interactive.session_ttl_s, config.interactive.max_turns)
    interactive = InteractiveSearch(
        smart, store, QueryRewriter(config.generation, config.interactive), FollowupSuggester(config.generation, config.interactive)
    )
    generator = httpx.Client(base_url=config.generation.url, timeout=5)
    checks |= {
        "qdrant": lambda: smart.retriever.dense.count() > 0,
        "valkey": store.ping,
        "generator": lambda: generator.get("/models").status_code == 200,
    }
    return Components(quick=quick, smart=smart, interactive=interactive, store=store, checks=checks)


def create_app(components: Components | None = None, config: ServingConfig | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.components = components or build_components(config or load_serving_config(), enabled_modes())
        yield

    app = FastAPI(title="Qimam Knowledge Bank Search", version="1.0.0", lifespan=lifespan)

    @app.exception_handler(httpx.HTTPError)
    async def generator_unavailable(request: Request, error: httpx.HTTPError) -> JSONResponse:
        logger.error("generator error: %s", error)
        return JSONResponse(status_code=503, content={"detail": "generator unavailable"})

    @app.exception_handler(SearchConnectionError)
    async def index_unavailable(request: Request, error: SearchConnectionError) -> JSONResponse:
        logger.error("opensearch error: %s", error)
        return JSONResponse(status_code=503, content={"detail": "search index unavailable"})

    @app.exception_handler(SessionConnectionError)
    async def sessions_unavailable(request: Request, error: SessionConnectionError) -> JSONResponse:
        logger.error("valkey error: %s", error)
        return JSONResponse(status_code=503, content={"detail": "session store unavailable"})

    @app.get("/livez")
    def livez() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health")
    def health(request: Request) -> JSONResponse:
        results = {}
        for name, check in request.app.state.components.checks.items():
            try:
                results[name] = "ok" if check() else "not_ready"
            except Exception as error:
                results[name] = f"error: {type(error).__name__}"
        healthy = all(value == "ok" for value in results.values())
        return JSONResponse(status_code=200 if healthy else 503, content={"status": "ok" if healthy else "degraded", "components": results})

    @app.post("/v1/search", response_model=QuickSearchResponse | SmartSearchResponse | InteractiveResponse)
    def search(body: SearchRequest, request: Request):
        parts: Components = request.app.state.components
        if body.mode == "quick_search":
            response = parts.quick.search(body.query, body.top_k)
        elif body.mode == "smart_ai_search" and parts.smart is not None:
            response = parts.smart.search(body.query)
        elif body.mode == "interactive" and parts.interactive is not None:
            response = parts.interactive.search(body.query, body.session_id)
        else:
            raise HTTPException(status_code=503, detail=f"{body.mode} is not enabled on this deployment")
        logger.info("mode=%s total_ms=%s status=%s", body.mode, response.latency_ms.get("total"), getattr(response, "status", "-"))
        return response

    def session_store(request: Request) -> SessionStore:
        store = request.app.state.components.store
        if store is None:
            raise HTTPException(status_code=503, detail="interactive is not enabled on this deployment")
        return store

    @app.get("/v1/sessions/{session_id}")
    def get_session(session_id: str, request: Request) -> dict:
        store = session_store(request)
        if not SESSION_ID_PATTERN.match(session_id):
            raise HTTPException(status_code=422, detail="invalid session id")
        turns = store.history(session_id)
        if not turns:
            raise HTTPException(status_code=404, detail="session not found or expired")
        return {"session_id": session_id, "ttl_s": store.ttl(session_id), "turns": [turn.model_dump() for turn in turns]}

    @app.delete("/v1/sessions/{session_id}")
    def delete_session(session_id: str, request: Request) -> dict:
        if not SESSION_ID_PATTERN.match(session_id):
            raise HTTPException(status_code=422, detail="invalid session id")
        return {"session_id": session_id, "deleted": session_store(request).clear(session_id)}

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(create_app(), host="0.0.0.0", port=int(os.environ.get("API_PORT", "8080")), workers=1)


if __name__ == "__main__":
    main()
