from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SERVING_CONFIG = REPO_ROOT / "configs" / "serving_config.yaml"


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    facts_dir: Path
    templates_dir: Path
    corpus_raw: Path
    corpus_processed: Path

    def resolved(self) -> PathsConfig:
        return PathsConfig(**{name: _resolve(value) for name, value in self.model_dump().items()})


class OpenSearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str
    index: str
    timeout_s: float = 10.0


class QdrantConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str
    collection: str
    timeout_s: float = 10.0


class ValkeyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str


class ChunkingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_chars: int = 1200


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())
    model: str
    device: str = "auto"
    batch_size: int = 16
    max_seq_length: int = 512
    query_instruction: str


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bm25_top_k: int = 50
    dense_top_k: int = 50
    rrf_k: int = 60
    bm25_weight: float = 1.0
    dense_weight: float = 1.0
    candidates: int = 30
    bm25_extra_candidates: int = 0


class RerankerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())
    model: str
    device: str = "auto"
    batch_size: int = 32
    max_length: int = 512


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())
    url: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 384
    timeout_s: float = 60.0


class SmartSearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    context_top_k: int = 4
    context_min_score: float = 0.0
    abstain_threshold: float = 0.0
    abstain_on_unsupported_numbers: bool = True


class InteractiveConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_ttl_s: int = 1800
    max_turns: int = 6
    rewrite_max_tokens: int = 96
    rewrite_prompt: str = "v1"
    rewrite_timeout_s: float = 20.0
    answer_context_chars: int = 400
    followups: int = 2
    followup_passages: int = 4
    followup_max_tokens: int = 96


class QuickSearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    top_k: int = 10
    snippet_chars: int = 240


class ServingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paths: PathsConfig
    opensearch: OpenSearchConfig
    qdrant: QdrantConfig
    valkey: ValkeyConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    quick_search: QuickSearchConfig
    reranker: RerankerConfig
    generation: GenerationConfig
    smart_search: SmartSearchConfig
    interactive: InteractiveConfig


_ENV_OVERRIDES = {
    "OPENSEARCH_URL": ("opensearch", "url"),
    "QDRANT_URL": ("qdrant", "url"),
    "VALKEY_URL": ("valkey", "url"),
    "EMBEDDING_MODEL": ("embedding", "model"),
    "QDRANT_COLLECTION": ("qdrant", "collection"),
    "RERANKER_MODEL": ("reranker", "model"),
    "VLLM_URL": ("generation", "url"),
    "REWRITE_PROMPT": ("interactive", "rewrite_prompt"),
}


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_serving_config(path: Path | None = None) -> ServingConfig:
    config_path = path or Path(os.environ.get("SERVING_CONFIG", DEFAULT_SERVING_CONFIG))
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for env_name, (section, key) in _ENV_OVERRIDES.items():
        value = os.environ.get(env_name)
        if value:
            raw[section][key] = value
    config = ServingConfig.model_validate(raw)
    return config.model_copy(update={"paths": config.paths.resolved()})
