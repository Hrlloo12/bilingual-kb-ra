from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import fitz

from rag.parsers.builder import BULLET_MARKERS, DocumentBuilder, as_bullet
from rag.schemas import ParsedDocument

HEADING_SIZE_RATIO = 1.15


@dataclass(frozen=True)
class _TextBlock:
    page: int
    text: str
    size: float


def _extract_blocks(document: fitz.Document) -> list[_TextBlock]:
    blocks: list[_TextBlock] = []
    for page_number, page in enumerate(document, start=1):
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            lines = []
            sizes = []
            for line in block["lines"]:
                text = "".join(span["text"] for span in line["spans"]).strip()
                if text:
                    lines.append(text)
                    sizes.extend(span["size"] for span in line["spans"] if span["text"].strip())
            if lines:
                blocks.append(_TextBlock(page=page_number, text=" ".join(lines), size=max(sizes)))
    return blocks


def _body_size(blocks: list[_TextBlock]) -> float:
    weights: Counter[float] = Counter()
    for block in blocks:
        weights[round(block.size, 1)] += len(block.text)
    return weights.most_common(1)[0][0]


def parse_pdf(path: Path) -> ParsedDocument:
    with fitz.open(str(path)) as document:
        blocks = _extract_blocks(document)
        page_count = document.page_count
    builder = DocumentBuilder("pdf", path.stem)
    if not blocks:
        return builder.build(page_count)
    body_size = _body_size(blocks)
    title_size = max(block.size for block in blocks)
    pending_bullet = False
    for block in blocks:
        if block.text in BULLET_MARKERS:
            pending_bullet = True
            continue
        if block.size >= body_size * HEADING_SIZE_RATIO:
            if builder.title is None and block.size >= title_size - 0.5:
                builder.set_title(block.text)
            else:
                builder.add_heading(block.text)
        elif pending_bullet or block.text.startswith(BULLET_MARKERS[:4]):
            builder.add_block(as_bullet(block.text), page=block.page)
        else:
            builder.add_block(block.text, page=block.page)
        pending_bullet = False
    return builder.build(page_count)
