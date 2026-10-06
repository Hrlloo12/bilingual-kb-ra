from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import fitz

from rag.langdetect import arabic_ratio
from rag.parsers.builder import BULLET_MARKERS, DocumentBuilder, as_bullet
from rag.schemas import ParsedDocument

HEADING_SIZE_RATIO = 1.15
RTL_DOCUMENT_RATIO = 0.3
ZERO_WIDTH_TOLERANCE = 0.05

_LEFT_TO_RIGHT_RUN = re.compile(
    "[A-Za-z0-9٠-٩%](?:[A-Za-z0-9٠-٩%.,:/\\-+&'’ ]*[A-Za-z0-9٠-٩%])?"
)
_ARABIC_LETTER = re.compile("[ء-ي]")
_SPACED_FATHATAN = re.compile("ً\\s+(?=ا)")
_MULTIPLE_SPACES = re.compile(r" {2,}")


@dataclass(frozen=True)
class _TextBlock:
    page: int
    text: str
    size: float


def restore_left_to_right_runs(right_to_left_reading: str) -> str:
    return _LEFT_TO_RIGHT_RUN.sub(lambda match: match.group()[::-1], right_to_left_reading)


def _glyph_clusters(characters: list[dict]) -> list[tuple[float, str]]:
    clusters: list[tuple[float, list[str]]] = []
    for character in characters:
        x0, _, x1, _ = character["bbox"]
        if clusters and x1 - x0 <= ZERO_WIDTH_TOLERANCE:
            clusters[-1][1].append(character["c"])
        else:
            clusters.append((x1, [character["c"]]))
    return [(right_edge, "".join(text)) for right_edge, text in clusters]


def _line_text(line: dict, right_to_left: bool) -> str:
    characters = [character for span in line["spans"] for character in span["chars"]]
    if right_to_left and any(_ARABIC_LETTER.match(character["c"]) for character in characters):
        clusters = _glyph_clusters(characters)
        ordered = sorted(enumerate(clusters), key=lambda item: (-item[1][0], item[0]))
        text = restore_left_to_right_runs("".join(text for _, (_, text) in ordered))
    else:
        text = "".join(character["c"] for character in characters)
    text = _SPACED_FATHATAN.sub("ً", text)
    return _MULTIPLE_SPACES.sub(" ", text).strip()


def _is_right_to_left(document: fitz.Document) -> bool:
    ratio = arabic_ratio("".join(page.get_text("text") for page in document))
    return ratio is not None and ratio >= RTL_DOCUMENT_RATIO


def _extract_blocks(document: fitz.Document) -> list[_TextBlock]:
    right_to_left = _is_right_to_left(document)
    blocks: list[_TextBlock] = []
    for page_number, page in enumerate(document, start=1):
        for block in page.get_text("rawdict")["blocks"]:
            if block.get("type") != 0:
                continue
            lines: list[str] = []
            sizes: list[float] = []
            for line in block["lines"]:
                text = _line_text(line, right_to_left)
                if text:
                    lines.append(text)
                    sizes.extend(span["size"] for span in line["spans"] if span["chars"])
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
