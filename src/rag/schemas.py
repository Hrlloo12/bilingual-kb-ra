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
