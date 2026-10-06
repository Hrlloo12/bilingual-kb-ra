from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from rag.parsers.builder import DocumentBuilder, as_bullet
from rag.schemas import ParsedDocument

_CONTENT_TAGS = ["h1", "h2", "h3", "h4", "p", "li"]
_REMOVED_TAGS = ["script", "style", "nav", "header", "footer", "noscript"]


def parse_html(path: Path) -> ParsedDocument:
    soup = BeautifulSoup(path.read_text(encoding="utf-8-sig"), "lxml")
    for tag in soup.find_all(_REMOVED_TAGS):
        tag.decompose()
    fallback_title = soup.title.get_text(" ", strip=True) if soup.title else path.stem
    builder = DocumentBuilder("html", fallback_title or path.stem)
    root = soup.body or soup
    for element in root.find_all(_CONTENT_TAGS):
        if element.find_parent("li") is not None:
            continue
        text = element.get_text(" ", strip=True)
        if element.name == "h1":
            builder.set_title(text)
        elif element.name in ("h2", "h3", "h4"):
            builder.add_heading(text)
        elif element.name == "li":
            builder.add_block(as_bullet(text))
        else:
            builder.add_block(text)
    return builder.build()
