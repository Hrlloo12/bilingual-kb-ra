from __future__ import annotations

from rag.normalize import count_script_letters
from rag.schemas import Language

ARABIC_DOMINANT_RATIO = 0.8
ENGLISH_DOMINANT_RATIO = 0.2


def arabic_ratio(text: str) -> float | None:
    arabic, latin = count_script_letters(text)
    total = arabic + latin
    return arabic / total if total else None


def detect_language(text: str) -> Language:
    ratio = arabic_ratio(text)
    if ratio is None or ratio <= ENGLISH_DOMINANT_RATIO:
        return "en"
    if ratio >= ARABIC_DOMINANT_RATIO:
        return "ar"
    return "mixed"
