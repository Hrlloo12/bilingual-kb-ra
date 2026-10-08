from __future__ import annotations

import json
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
    first_token_ms: float | None = None


def read_stream(lines) -> tuple[str, dict, float | None, float]:
    started = perf_counter()
    parts: list[str] = []
    usage: dict = {}
    first_token: float | None = None
    for line in lines:
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        event = json.loads(data)
        usage = event.get("usage") or usage
        for choice in event.get("choices") or []:
            text = (choice.get("delta") or {}).get("content")
            if text:
                if first_token is None:
                    first_token = perf_counter()
                parts.append(text)
    first_token_ms = round((first_token - started) * 1000, 2) if first_token is not None else None
    return "".join(parts), usage, first_token_ms, round((perf_counter() - started) * 1000, 2)


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
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        started = perf_counter()
        with self.client.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()
            connected_ms = (perf_counter() - started) * 1000
            text, usage, first_token_ms, streamed_ms = read_stream(response.iter_lines())
        return Generation(
            text=text.strip(),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=round(connected_ms + streamed_ms, 2),
            first_token_ms=round(connected_ms + first_token_ms, 2) if first_token_ms is not None else None,
        )
