from __future__ import annotations

from rag.normalize import normalize_display
from rag.schemas import DocumentFormat, ParsedBlock, ParsedDocument, ParsedSection

BULLET_MARKERS = ("•", "◦", "▪", "●", "-", "*")


def as_bullet(text: str) -> str:
    stripped = text.strip()
    for marker in BULLET_MARKERS:
        if stripped.startswith(marker):
            stripped = stripped[len(marker):].strip()
            break
    return f"- {stripped}"


class DocumentBuilder:
    def __init__(self, document_format: DocumentFormat, fallback_title: str) -> None:
        self.format = document_format
        self.fallback_title = fallback_title
        self.title: str | None = None
        self.sections: list[ParsedSection] = [ParsedSection(heading=None)]

    def set_title(self, text: str) -> None:
        text = normalize_display(text)
        if not text:
            return
        if self.title is None:
            self.title = text
        else:
            self.add_heading(text)

    def add_heading(self, text: str) -> None:
        text = normalize_display(text)
        if text:
            self.sections.append(ParsedSection(heading=text))

    def add_block(self, text: str, page: int | None = None) -> None:
        text = normalize_display(text)
        if text:
            self.sections[-1].blocks.append(ParsedBlock(text=text, page=page))

    def build(self, page_count: int | None = None) -> ParsedDocument:
        sections = [section for section in self.sections if section.blocks]
        return ParsedDocument(
            title=self.title or self.fallback_title,
            format=self.format,
            page_count=page_count,
            sections=sections,
        )
