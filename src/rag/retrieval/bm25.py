from __future__ import annotations

from dataclasses import dataclass

from opensearchpy import OpenSearch, helpers

from rag.config import OpenSearchConfig
from rag.schemas import Chunk

INDEX_SETTINGS = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "filter": {
                "arabic_stop": {"type": "stop", "stopwords": "_arabic_"},
                "english_stop": {"type": "stop", "stopwords": "_english_"},
                "english_possessive": {"type": "stemmer", "language": "possessive_english"},
                "english_stemmer": {"type": "stemmer", "language": "english"},
            },
            "analyzer": {
                "bilingual": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                        "decimal_digit",
                        "arabic_normalization",
                        "arabic_stop",
                        "arabic_stem",
                        "english_possessive",
                        "english_stop",
                        "english_stemmer",
                    ],
                }
            },
        },
    },
    "mappings": {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "position": {"type": "integer"},
            "title": {"type": "text", "analyzer": "bilingual"},
            "section": {"type": "text", "analyzer": "bilingual"},
            "text": {"type": "text", "analyzer": "bilingual"},
            "page": {"type": "integer"},
            "language": {"type": "keyword"},
            "domain": {"type": "keyword"},
            "format": {"type": "keyword"},
            "source": {"type": "keyword"},
        }
    },
}

SEARCH_FIELDS = ["title^1.5", "section^2", "text"]


@dataclass(frozen=True)
class LexicalHit:
    chunk: Chunk
    score: float
    snippet: str


class BM25Index:
    def __init__(self, config: OpenSearchConfig) -> None:
        self.index = config.index
        self.client = OpenSearch(hosts=[config.url], timeout=config.timeout_s, http_compress=True)

    def recreate(self) -> None:
        if self.client.indices.exists(index=self.index):
            self.client.indices.delete(index=self.index)
        self.client.indices.create(index=self.index, body=INDEX_SETTINGS)

    def ensure_exists(self) -> None:
        if not self.client.indices.exists(index=self.index):
            self.client.indices.create(index=self.index, body=INDEX_SETTINGS)

    def delete_documents(self, doc_ids: list[str]) -> None:
        if doc_ids:
            self.client.delete_by_query(
                index=self.index, body={"query": {"terms": {"doc_id": doc_ids}}}, refresh=True, conflicts="proceed"
            )

    def add(self, chunks: list[Chunk]) -> int:
        actions = ({"_index": self.index, "_id": chunk.chunk_id, "_source": chunk.model_dump()} for chunk in chunks)
        indexed, _ = helpers.bulk(self.client, actions, refresh="wait_for")
        return indexed

    def count(self) -> int:
        return self.client.count(index=self.index)["count"]

    def search(self, query: str, top_k: int, snippet_chars: int = 240) -> list[LexicalHit]:
        body = {
            "size": top_k,
            "query": {
                "multi_match": {"query": query, "fields": SEARCH_FIELDS, "type": "best_fields", "tie_breaker": 0.3}
            },
            "highlight": {
                "pre_tags": [""],
                "post_tags": [""],
                "fields": {"text": {"fragment_size": snippet_chars, "number_of_fragments": 1, "no_match_size": snippet_chars}},
            },
        }
        response = self.client.search(index=self.index, body=body)
        hits = []
        for hit in response["hits"]["hits"]:
            fragments = hit.get("highlight", {}).get("text", [])
            snippet = fragments[0] if fragments else hit["_source"]["text"][:snippet_chars]
            hits.append(LexicalHit(chunk=Chunk.model_validate(hit["_source"]), score=hit["_score"], snippet=snippet))
        return hits
