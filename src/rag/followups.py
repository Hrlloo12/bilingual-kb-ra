from __future__ import annotations

import re

import httpx

from rag.config import GenerationConfig, InteractiveConfig
from rag.generation import format_passages
from rag.langdetect import detect_language
from rag.normalize import normalize_for_search
from rag.schemas import Chunk, Language

FOLLOWUP_PROMPT = """You suggest follow-up questions for a company knowledge-bank search.
Rules:
1. Write exactly {count} short questions the user could ask next.
2. Each question must be answerable from the numbered passages, and must ask something different from the user's question.
3. Name the product, place, service or policy explicitly instead of using pronouns.
4. Write every question in {language}.
Output one question per line, with no numbering and nothing else."""

FOLLOWUP_LANGUAGES: dict[Language, str] = {
    "ar": "Modern Standard Arabic",
    "en": "English",
    "mixed": "Arabic, keeping English product names in English",
}

_LIST_PREFIX = re.compile(r"^\s*(?:[-*•]|\d+[.)]|[٠-٩]+[.)])\s*")


def build_followup_messages(query: str, language: Language, chunks: list[Chunk], count: int) -> list[dict[str, str]]:
    system = FOLLOWUP_PROMPT.format(count=count, language=FOLLOWUP_LANGUAGES[language])
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Passages:\n\n{format_passages(chunks)}\n\nUser question: {query}\n\nFollow-up questions:"},
    ]


def parse_followups(text: str, query: str, language: Language, count: int) -> list[str]:
    asked = normalize_for_search(query)
    questions: list[str] = []
    for line in text.splitlines():
        question = _LIST_PREFIX.sub("", line).strip().strip("\"'«»“”`").strip()
        if len(question) < 6 or normalize_for_search(question) == asked:
            continue
        detected = detect_language(question)
        if (language == "en") != (detected == "en"):
            continue
        if normalize_for_search(question) in {normalize_for_search(item) for item in questions}:
            continue
        questions.append(question)
        if len(questions) == count:
            break
    return questions


class FollowupSuggester:
    def __init__(self, generation: GenerationConfig, settings: InteractiveConfig) -> None:
        self.generation = generation
        self.settings = settings
        self.client = httpx.Client(base_url=generation.url, timeout=settings.rewrite_timeout_s)

    def complete(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.generation.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.settings.followup_max_tokens,
            "seed": 0,
        }
        response = self.client.post("/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def suggest(self, query: str, language: Language, chunks: list[Chunk]) -> list[str]:
        count = self.settings.followups
        if count <= 0 or not chunks:
            return []
        try:
            text = self.complete(build_followup_messages(query, language, chunks[: self.settings.followup_passages], count))
        except httpx.HTTPError:
            return []
        return parse_followups(text, query, language, count)
