from __future__ import annotations

import json
import re

import httpx

from rag.grounding import strip_markers
from rag.normalize import normalize_for_search

_TOKEN = re.compile(r"\w+")
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

JUDGE_PROMPT = """You are a strict evaluator for a retrieval-augmented question answering system over a company knowledge bank.
You receive a question, the numbered source passages the system cited, and the system's answer. The answer may be in a different language from the passages; translation between Arabic and English is allowed and is not an error.
Judge two things:
1. supported: true only if every factual claim in the answer (numbers, prices, dates, durations, names, places, conditions) is stated in or directly implied by the passages.
2. relevance: "full" if the answer gives what the question asks, "partial" if it answers only part or a related but different attribute, "none" if it does not address the question.
Reply with JSON only: {"supported": true or false, "unsupported_claims": ["..."], "relevance": "full" or "partial" or "none"}"""


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(normalize_for_search(strip_markers(text)))


def _lcs_length(first: list[str], second: list[str]) -> int:
    previous = [0] * (len(second) + 1)
    for item in first:
        current = [0]
        for index, other in enumerate(second, start=1):
            current.append(previous[index - 1] + 1 if item == other else max(previous[index], current[index - 1]))
        previous = current
    return previous[-1]


def rouge_l(candidate: str, reference: str) -> float:
    candidate_tokens, reference_tokens = tokens(candidate), tokens(reference)
    if not candidate_tokens or not reference_tokens:
        return 0.0
    overlap = _lcs_length(candidate_tokens, reference_tokens)
    if not overlap:
        return 0.0
    precision_value, recall_value = overlap / len(candidate_tokens), overlap / len(reference_tokens)
    return 2 * precision_value * recall_value / (precision_value + recall_value)


def fact_coverage(chunk_ids: list[str], labels: dict[str, frozenset[str]], targets: set[str]) -> float:
    if not targets:
        return 0.0
    covered = set().union(*(labels.get(chunk_id, frozenset()) for chunk_id in chunk_ids)) if chunk_ids else set()
    return len(covered & targets) / len(targets)


def precision(chunk_ids: list[str], labels: dict[str, frozenset[str]], targets: set[str]) -> float | None:
    if not chunk_ids:
        return None
    return sum(1 for chunk_id in chunk_ids if labels.get(chunk_id, frozenset()) & targets) / len(chunk_ids)


def parse_verdict(text: str) -> dict | None:
    match = _JSON_OBJECT.search(text)
    if not match:
        return None
    try:
        verdict = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(verdict.get("supported"), bool) or verdict.get("relevance") not in ("full", "partial", "none"):
        return None
    claims = verdict.get("unsupported_claims") or []
    return {"supported": verdict["supported"], "relevance": verdict["relevance"], "unsupported_claims": [str(claim) for claim in claims]}


class Judge:
    def __init__(self, url: str, model: str, timeout_s: float = 120.0) -> None:
        self.model = model
        self.client = httpx.Client(base_url=url, timeout=timeout_s)

    def judge(self, question: str, passages: list[str], answer: str) -> dict | None:
        numbered = "\n\n".join(f"[{number}] {passage}" for number, passage in enumerate(passages, start=1))
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": f"Question: {question}\n\nPassages:\n{numbered}\n\nAnswer: {strip_markers(answer)}"},
            ],
            "temperature": 0.0,
            "max_tokens": 256,
            "seed": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        response = self.client.post("/chat/completions", json=payload)
        response.raise_for_status()
        return parse_verdict(response.json()["choices"][0]["message"]["content"])
