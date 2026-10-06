from __future__ import annotations

from pathlib import Path

from rag.parsers.builder import DocumentBuilder, as_bullet
from rag.schemas import ParsedDocument


def parse_text(path: Path) -> ParsedDocument:
    builder = DocumentBuilder("txt", path.stem)
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            builder.add_block(" ".join(paragraph))
            paragraph.clear()

    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line:
            flush()
        elif line.startswith("## "):
            flush()
            builder.add_heading(line[3:])
        elif line.startswith("# "):
            flush()
            builder.set_title(line[2:])
        elif line.startswith("- "):
            flush()
            builder.add_block(as_bullet(line))
        else:
            paragraph.append(line)
    flush()
    return builder.build()
