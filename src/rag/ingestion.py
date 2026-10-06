from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from rag.chunking import chunk_document
from rag.config import ServingConfig
from rag.langdetect import detect_language
from rag.parsers import is_supported, parse_file
from rag.retrieval.bm25 import BM25Index
from rag.schemas import Chunk, DocumentFormat, DocumentMetadata

MANIFEST_NAME = "manifest.jsonl"
CHUNKS_FILE_NAME = "chunks.jsonl"
_DOC_ID_UNSAFE = re.compile(r"[^a-z0-9_]+")
_FORMAT_BY_SUFFIX: dict[str, DocumentFormat] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".html": "html",
    ".htm": "html",
    ".txt": "txt",
    ".md": "txt",
}


@dataclass
class IngestionReport:
    documents: int = 0
    chunks: int = 0
    by_language: Counter = field(default_factory=Counter)
    by_format: Counter = field(default_factory=Counter)
    failures: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "documents": self.documents,
            "chunks": self.chunks,
            "by_language": dict(self.by_language),
            "by_format": dict(self.by_format),
            "failures": self.failures,
        }


def discover(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if is_supported(path) else []
    return sorted(candidate for candidate in path.rglob("*") if candidate.is_file() and is_supported(candidate))


def load_manifest(root: Path) -> dict[str, dict]:
    manifest = root / MANIFEST_NAME
    if not manifest.exists():
        return {}
    records = (json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip())
    return {record["doc_id"]: record for record in records}


def make_doc_id(path: Path) -> str:
    return _DOC_ID_UNSAFE.sub("_", path.stem.lower()).strip("_")


def parse_and_chunk(paths: list[Path], root: Path, max_chars: int) -> tuple[list[Chunk], IngestionReport]:
    manifest = load_manifest(root)
    report = IngestionReport()
    chunks: list[Chunk] = []
    seen: dict[str, Path] = {}
    for path in paths:
        doc_id = make_doc_id(path)
        if doc_id in seen:
            raise ValueError(f"duplicate doc_id {doc_id} for {path} and {seen[doc_id]}")
        seen[doc_id] = path
        try:
            document = parse_file(path)
        except Exception as error:
            report.failures[str(path)] = f"{type(error).__name__}: {error}"
            continue
        relative = path.relative_to(root) if path.is_relative_to(root) else Path(path.name)
        record = manifest.get(doc_id)
        metadata = DocumentMetadata(
            doc_id=doc_id,
            title=document.title,
            language=record["language"] if record else detect_language(document.full_text),
            domain=relative.parts[0] if len(relative.parts) > 1 else "general",
            format=_FORMAT_BY_SUFFIX[path.suffix.lower()],
            source=relative.as_posix(),
        )
        document_chunks = chunk_document(document, metadata, max_chars)
        chunks.extend(document_chunks)
        report.documents += 1
        report.chunks += len(document_chunks)
        report.by_language[metadata.language] += 1
        report.by_format[metadata.format] += 1
    return chunks, report


def read_chunks(path: Path) -> list[Chunk]:
    if not path.exists():
        return []
    return [Chunk.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(chunk.model_dump_json() + "\n")


class KnowledgeBankIndexer:
    def __init__(self, config: ServingConfig, with_dense: bool = True) -> None:
        self.config = config
        self.bm25 = BM25Index(config.opensearch)
        self.with_dense = with_dense
        self.embedder = None
        self.dense = None
        if with_dense:
            from rag.retrieval.dense import DenseIndex, Embedder

            self.embedder = Embedder(config.embedding)
            self.dense = DenseIndex(config.qdrant)

    @property
    def chunks_path(self) -> Path:
        return self.config.paths.corpus_processed / CHUNKS_FILE_NAME

    def rebuild(self, chunks: list[Chunk]) -> dict[str, int]:
        self.bm25.recreate()
        counts = {"bm25": self.bm25.add(chunks)}
        if self.with_dense:
            self.dense.recreate(self.embedder.dimension)
            counts["dense"] = self.dense.add(chunks, self.embedder.encode_passages(chunks))
        write_chunks(chunks, self.chunks_path)
        return counts

    def upsert(self, chunks: list[Chunk]) -> dict[str, int]:
        doc_ids = sorted({chunk.doc_id for chunk in chunks})
        self.bm25.ensure_exists()
        self.bm25.delete_documents(doc_ids)
        counts = {"bm25": self.bm25.add(chunks)}
        if self.with_dense:
            self.dense.ensure_exists(self.embedder.dimension)
            self.dense.delete_documents(doc_ids)
            counts["dense"] = self.dense.add(chunks, self.embedder.encode_passages(chunks))
        retained = [chunk for chunk in read_chunks(self.chunks_path) if chunk.doc_id not in set(doc_ids)]
        write_chunks(retained + chunks, self.chunks_path)
        return counts
