from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from rag.normalize import search_key
from rag.schemas import Chunk, Language


class RetrievalQuery(BaseModel):
    id: str
    query: str
    language: Language
    bucket: str
    relevant_fact_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @property
    def answerable(self) -> bool:
        return bool(self.relevant_fact_ids)


def load_queries(path: Path) -> list[RetrievalQuery]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [RetrievalQuery.model_validate_json(line) for line in lines if line.strip()]


def chunk_fact_labels(chunks: list[Chunk], manifest_path: Path) -> dict[str, frozenset[str]]:
    section_facts: dict[tuple[str, str], list[str]] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        for section in record["sections"]:
            section_facts[(record["doc_id"], search_key(section["heading"]))] = section["fact_ids"]
    labels: dict[str, frozenset[str]] = {}
    matched_sections: set[tuple[str, str]] = set()
    for chunk in chunks:
        key = (chunk.doc_id, search_key(chunk.section or ""))
        if key in section_facts:
            matched_sections.add(key)
        labels[chunk.chunk_id] = frozenset(section_facts.get(key, []))
    unmatched = sorted(set(section_facts) - matched_sections)
    if unmatched:
        raise ValueError(f"manifest sections without chunks: {unmatched[:5]}")
    return labels


def relevant_chunk_ids(query: RetrievalQuery, labels: dict[str, frozenset[str]]) -> set[str]:
    targets = set(query.relevant_fact_ids)
    return {chunk_id for chunk_id, facts in labels.items() if facts & targets}
