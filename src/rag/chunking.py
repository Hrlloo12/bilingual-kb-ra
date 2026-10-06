from __future__ import annotations

import re

from rag.schemas import Chunk, DocumentMetadata, ParsedBlock, ParsedDocument

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?؟])\s+")


def _split_oversized(block: ParsedBlock, max_chars: int) -> list[ParsedBlock]:
    if len(block.text) <= max_chars:
        return [block]
    pieces: list[ParsedBlock] = []
    current = ""
    for sentence in _SENTENCE_BOUNDARY.split(block.text):
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_chars:
            pieces.append(ParsedBlock(text=current, page=block.page))
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(ParsedBlock(text=current, page=block.page))
    return pieces


def _group_blocks(blocks: list[ParsedBlock], max_chars: int) -> list[list[ParsedBlock]]:
    groups: list[list[ParsedBlock]] = []
    current: list[ParsedBlock] = []
    size = 0
    for block in (piece for block in blocks for piece in _split_oversized(block, max_chars)):
        if current and size + len(block.text) + 1 > max_chars:
            groups.append(current)
            current, size = [], 0
        current.append(block)
        size += len(block.text) + 1
    if current:
        groups.append(current)
    return groups


def chunk_document(document: ParsedDocument, metadata: DocumentMetadata, max_chars: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in document.sections:
        for group in _group_blocks(section.blocks, max_chars):
            position = len(chunks) + 1
            chunks.append(
                Chunk(
                    chunk_id=f"{metadata.doc_id}#{position}",
                    doc_id=metadata.doc_id,
                    position=position,
                    title=metadata.title,
                    section=section.heading,
                    page=next((block.page for block in group if block.page is not None), None),
                    language=metadata.language,
                    domain=metadata.domain,
                    format=metadata.format,
                    source=metadata.source,
                    text="\n".join(block.text for block in group),
                )
            )
    return chunks
