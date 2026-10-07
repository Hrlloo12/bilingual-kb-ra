from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Language = Literal["ar", "en", "mixed"]
DocumentFormat = Literal["pdf", "docx", "html", "txt"]


class ParsedBlock(BaseModel):
    model_config = ConfigDict(frozen=True)
    text: str
    page: int | None = None


class ParsedSection(BaseModel):
    heading: str | None
    blocks: list[ParsedBlock] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    title: str
    format: DocumentFormat
    page_count: int | None = None
    sections: list[ParsedSection]

    @property
    def full_text(self) -> str:
        parts = [self.title]
        for section in self.sections:
            if section.heading:
                parts.append(section.heading)
            parts.extend(block.text for block in section.blocks)
        return "\n".join(parts)


class DocumentMetadata(BaseModel):
    doc_id: str
    title: str
    language: Language
    domain: str
    format: DocumentFormat
    source: str


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    position: int
    title: str
    section: str | None
    page: int | None
    language: Language
    domain: str
    format: DocumentFormat
    source: str
    text: str

    @property
    def context_header(self) -> str:
        return f"{self.title} | {self.section}" if self.section else self.title


class QuickSearchResult(BaseModel):
    doc_id: str
    chunk_id: str
    title: str
    section: str | None
    snippet: str
    score: float
    source: str
    page: int | None


class QuickSearchResponse(BaseModel):
    mode: Literal["quick_search"] = "quick_search"
    query: str
    language_detected: Language
    results: list[QuickSearchResult]
    latency_ms: dict[str, float]


class Citation(BaseModel):
    marker: int
    doc_id: str
    chunk_id: str
    title: str
    section: str | None
    page: int | None
    source: str
    language: Language


class RetrievedPassage(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    section: str | None
    rerank_score: float | None
    fused_score: float
    bm25_rank: int | None
    dense_rank: int | None
    in_context: bool


class SmartSearchResponse(BaseModel):
    mode: Literal["smart_search"] = "smart_search"
    query: str
    language_detected: Language
    status: Literal["answered", "not_found"]
    answer: str
    citations: list[Citation]
    abstain_reason: str | None = None
    post_checks: dict = Field(default_factory=dict)
    retrieved: list[RetrievedPassage]
    usage: dict[str, int] = Field(default_factory=dict)
    latency_ms: dict[str, float]


class RewriteInfo(BaseModel):
    applied: bool
    reason: str
    standalone_query: str
    latency_ms: float
    fallback: bool = False


class InteractiveResponse(BaseModel):
    mode: Literal["interactive"] = "interactive"
    session_id: str
    turn: int
    new_session: bool
    query: str
    rewrite: RewriteInfo
    result: SmartSearchResponse
    latency_ms: dict[str, float]
