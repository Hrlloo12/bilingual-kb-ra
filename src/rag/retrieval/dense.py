from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np

from rag.config import EmbeddingConfig, QdrantConfig
from rag.normalize import normalize_for_dense
from rag.schemas import Chunk

_POINT_NAMESPACE = uuid.UUID("6f1c2d3e-4b5a-4c6d-8e7f-9a0b1c2d3e4f")


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


def passage_text(chunk: Chunk) -> str:
    return normalize_for_dense(f"{chunk.context_header}\n{chunk.text}")


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class Embedder:
    def __init__(self, config: EmbeddingConfig) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self.config = config
        self.device = resolve_device(config.device)
        model_kwargs = {"torch_dtype": torch.float16} if self.device == "cuda" else {}
        self.model = SentenceTransformer(config.model, device=self.device, model_kwargs=model_kwargs)
        self.model.max_seq_length = config.max_seq_length
        self.query_prompt = f"Instruct: {config.query_instruction}\nQuery:"

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        texts = [normalize_for_dense(query) for query in queries]
        return self.model.encode(
            texts, prompt=self.query_prompt, batch_size=self.config.batch_size, normalize_embeddings=True
        )

    def encode_passages(self, chunks: list[Chunk]) -> np.ndarray:
        return self.model.encode(
            [passage_text(chunk) for chunk in chunks],
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(chunks) > 64,
        )


@dataclass(frozen=True)
class DenseHit:
    chunk: Chunk
    score: float


class DenseIndex:
    def __init__(self, config: QdrantConfig) -> None:
        from qdrant_client import QdrantClient

        self.collection = config.collection
        self.client = QdrantClient(url=config.url, timeout=int(config.timeout_s))

    def recreate(self, dimension: int) -> None:
        from qdrant_client import models

        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
        )

    def ensure_exists(self, dimension: int) -> None:
        if not self.client.collection_exists(self.collection):
            self.recreate(dimension)

    def delete_documents(self, doc_ids: list[str]) -> None:
        from qdrant_client import models

        if doc_ids:
            self.client.delete(
                self.collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=doc_ids))])
                ),
                wait=True,
            )

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        from qdrant_client import models

        points = [
            models.PointStruct(id=point_id(chunk.chunk_id), vector=vector.tolist(), payload=chunk.model_dump())
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(self.collection, points=points, wait=True)
        return len(points)

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def search(self, vector: np.ndarray, top_k: int) -> list[DenseHit]:
        response = self.client.query_points(self.collection, query=vector.tolist(), limit=top_k, with_payload=True)
        return [DenseHit(chunk=Chunk.model_validate(point.payload), score=point.score) for point in response.points]
