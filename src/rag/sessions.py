from __future__ import annotations

import re
import time
import uuid

from pydantic import BaseModel, Field

from rag.schemas import Language

SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
KEY_PREFIX = "kb:session:"


class Turn(BaseModel):
    query: str
    standalone_query: str
    language: Language
    status: str
    answer: str
    cited_chunk_ids: list[str] = Field(default_factory=list)
    timestamp: float = Field(default_factory=time.time)


def new_session_id() -> str:
    return uuid.uuid4().hex


def valid_session_id(session_id: str) -> bool:
    return bool(SESSION_ID_PATTERN.match(session_id))


class SessionStore:
    def __init__(self, client, ttl_s: int, max_turns: int) -> None:
        self.client = client
        self.ttl_s = ttl_s
        self.max_turns = max_turns

    @classmethod
    def from_url(cls, url: str, ttl_s: int, max_turns: int) -> SessionStore:
        import valkey

        return cls(valkey.Valkey.from_url(url, decode_responses=True, socket_timeout=5), ttl_s, max_turns)

    def _key(self, session_id: str) -> str:
        if not valid_session_id(session_id):
            raise ValueError("invalid session id")
        return KEY_PREFIX + session_id

    def history(self, session_id: str) -> list[Turn]:
        return [Turn.model_validate_json(item) for item in self.client.lrange(self._key(session_id), 0, -1)]

    def append(self, session_id: str, turn: Turn) -> None:
        key = self._key(session_id)
        pipeline = self.client.pipeline()
        pipeline.rpush(key, turn.model_dump_json())
        pipeline.ltrim(key, -self.max_turns, -1)
        pipeline.expire(key, self.ttl_s)
        pipeline.execute()

    def ttl(self, session_id: str) -> int:
        return int(self.client.ttl(self._key(session_id)))

    def clear(self, session_id: str) -> bool:
        return bool(self.client.delete(self._key(session_id)))

    def ping(self) -> bool:
        return bool(self.client.ping())
