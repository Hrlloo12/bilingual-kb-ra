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


class ServingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paths: PathsConfig
    opensearch: OpenSearchConfig
    qdrant: QdrantConfig
    valkey: ValkeyConfig


_ENV_OVERRIDES = {
    "OPENSEARCH_URL": ("opensearch", "url"),
    "QDRANT_URL": ("qdrant", "url"),
    "VALKEY_URL": ("valkey", "url"),
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
