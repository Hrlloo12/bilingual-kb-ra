from __future__ import annotations

from rag.config import RerankerConfig
from rag.normalize import normalize_for_dense
from rag.retrieval.dense import passage_text, resolve_device
from rag.retrieval.hybrid import Candidate


class Reranker:
    def __init__(self, config: RerankerConfig) -> None:
        import torch
        from sentence_transformers.cross_encoder import CrossEncoder

        self.config = config
        self.device = resolve_device(config.device)
        model_kwargs = {"torch_dtype": torch.float16} if self.device == "cuda" else {}
        self.model = CrossEncoder(config.model, max_length=config.max_length, device=self.device, model_kwargs=model_kwargs)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        pairs = [(normalize_for_dense(query), passage) for passage in passages]
        scores = self.model.predict(pairs, batch_size=self.config.batch_size, show_progress_bar=False, convert_to_numpy=True)
        return [float(score) for score in scores]

    def rerank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        scores = self.score(query, [passage_text(candidate.chunk) for candidate in candidates])
        for candidate, score in zip(candidates, scores, strict=True):
            candidate.rerank_score = score
        return sorted(candidates, key=lambda candidate: -candidate.rerank_score)
