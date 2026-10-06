from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from rag.schemas import ParsedDocument


def _parse_pdf(path: Path) -> ParsedDocument:
    from rag.parsers.pdf_parser import parse_pdf

    return parse_pdf(path)


def _parse_docx(path: Path) -> ParsedDocument:
    from rag.parsers.docx_parser import parse_docx

    return parse_docx(path)


def _parse_html(path: Path) -> ParsedDocument:
    from rag.parsers.html_parser import parse_html

    return parse_html(path)


def _parse_text(path: Path) -> ParsedDocument:
    from rag.parsers.text_parser import parse_text

    return parse_text(path)


PARSERS: dict[str, Callable[[Path], ParsedDocument]] = {
    ".pdf": _parse_pdf,
    ".docx": _parse_docx,
    ".html": _parse_html,
    ".htm": _parse_html,
    ".txt": _parse_text,
    ".md": _parse_text,
}


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in PARSERS


def parse_file(path: Path) -> ParsedDocument:
    suffix = path.suffix.lower()
    if suffix not in PARSERS:
        raise ValueError(f"unsupported document type {suffix} for {path}")
    return PARSERS[suffix](path)
