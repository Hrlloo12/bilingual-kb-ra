from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import httpx

from rag.config import GenerationConfig
from rag.schemas import Chunk, Language

NOT_FOUND = "NOT_FOUND"

SYSTEM_PROMPT = """You answer employee and customer questions about Qimam Home using only the numbered passages you are given.
Rules:
1. Use only facts stated in the passages. Never add outside knowledge or guess.
2. End every sentence with the number of the passage that supports it, in square brackets, for example [1] or [2][3].
3. Copy numbers, prices, dates, durations, phone numbers, product names and codes exactly as the passages state them.
4. Answer in {language}.
5. Answer only what was asked, in one to three sentences.
6. If the passages do not contain the information needed to answer, reply with exactly NOT_FOUND and nothing else."""

LANGUAGE_INSTRUCTIONS: dict[Language, str] = {
    "ar": "Modern Standard Arabic, even when the passages are in English",
    "en": "English, even when the passages are in Arabic",
    "mixed": "Arabic, keeping English product names and technical terms in English",
}


def format_passages(chunks: list[Chunk]) -> str:
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        blocks.append(f"[{number}] {chunk.context_header}\n{chunk.text}")
    return "\n\n".join(blocks)


def build_messages(query: str, language: Language, chunks: list[Chunk]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(language=LANGUAGE_INSTRUCTIONS[language])},
        {"role": "user", "content": f"Passages:\n\n{format_passages(chunks)}\n\nQuestion: {query}"},
    ]


@dataclass(frozen=True)
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class Generator:
    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self.client = httpx.Client(base_url=config.url, timeout=config.timeout_s)

    def generate(self, query: str, language: Language, chunks: list[Chunk]) -> Generation:
        payload = {
            "model": self.config.model,
            "messages": build_messages(query, language, chunks),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "seed": 0,
        }
        started = perf_counter()
        response = self.client.post("/chat/completions", json=payload)
        response.raise_for_status()
        latency = round((perf_counter() - started) * 1000, 2)
        body = response.json()
        usage = body.get("usage") or {}
        return Generation(
            text=body["choices"][0]["message"]["content"].strip(),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency,
        )
