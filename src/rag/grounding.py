from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag.generation import NOT_FOUND
from rag.langdetect import detect_language
from rag.normalize import extract_numbers, fold_digits, normalize_for_search
from rag.schemas import Chunk, Language

_MARKER = re.compile(r"\[\s*([0-9٠-٩]+(?:\s*[,،]\s*[0-9٠-٩]+)*)\s*\]")
_CODE = re.compile(r"\b(?=[A-Za-z0-9-]*\d)(?=[A-Za-z0-9-]*[A-Za-z])[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b")


@dataclass
class GroundedAnswer:
    text: str
    cited: list[int]
    not_found: bool
    checks: dict = field(default_factory=dict)


def parse_markers(answer: str, passage_count: int) -> tuple[list[int], list[int]]:
    cited, invalid = [], []
    for match in _MARKER.finditer(answer):
        for value in re.split(r"[,،]", fold_digits(match.group(1))):
            number = int(value.strip())
            target = cited if 1 <= number <= passage_count else invalid
            if number not in target:
                target.append(number)
    return cited, invalid


def strip_markers(answer: str) -> str:
    return re.sub(r"\s+([.،,؟?!])", r"\1", _MARKER.sub("", answer)).strip()


def is_not_found(answer: str) -> bool:
    return NOT_FOUND in answer.upper().replace(" ", "_")


def unsupported_numbers(answer: str, evidence: list[str]) -> list[float]:
    allowed = {value for text in evidence for value in extract_numbers(text)}
    return sorted({value for value in extract_numbers(answer) if value not in allowed})


def unsupported_codes(answer: str, evidence: list[str]) -> list[str]:
    haystack = normalize_for_search(" ".join(evidence))
    return sorted({code for code in _CODE.findall(answer) if normalize_for_search(code) not in haystack})


def language_matches(answer: str, query_language: Language) -> bool:
    detected = detect_language(answer)
    if query_language == "en":
        return detected == "en"
    return detected in ("ar", "mixed")


def ground(answer: str, query: str, query_language: Language, chunks: list[Chunk]) -> GroundedAnswer:
    if is_not_found(answer):
        return GroundedAnswer(text=NOT_FOUND, cited=[], not_found=True, checks={"generator_not_found": True})
    cited, invalid = parse_markers(answer, len(chunks))
    plain = strip_markers(answer)
    evidence_chunks = [chunks[number - 1] for number in cited] or chunks
    evidence = [f"{chunk.context_header}\n{chunk.text}" for chunk in evidence_chunks] + [query]
    checks = {
        "generator_not_found": False,
        "has_citation": bool(cited),
        "invalid_citations": invalid,
        "unsupported_numbers": unsupported_numbers(plain, evidence),
        "unsupported_codes": unsupported_codes(plain, evidence),
        "language_ok": language_matches(plain, query_language),
    }
    return GroundedAnswer(text=answer.strip(), cited=cited, not_found=False, checks=checks)
