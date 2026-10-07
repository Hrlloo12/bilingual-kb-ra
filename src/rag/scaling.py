from __future__ import annotations

import random
import re

from rag.schemas import Chunk

_LATIN_NAME = re.compile(r"\b[A-Z][a-z]{2,}\b")
_NUMBER = re.compile(r"\d+")
_SYLLABLES = ("ka", "lo", "mi", "ra", "su", "ne", "ti", "va", "zo", "ha", "di", "ru", "sa", "me", "fa", "lu")


def _name(rng: random.Random) -> str:
    return "".join(rng.choice(_SYLLABLES) for _ in range(rng.randint(2, 3))).capitalize()


def _digits(match: re.Match, rng: random.Random) -> str:
    width = len(match.group(0))
    low = 10 ** (width - 1) if width > 1 else 0
    return str(rng.randint(low, 10**width - 1))


def make_distractors(chunks: list[Chunk], count: int, seed: int = 2026) -> list[Chunk]:
    rng = random.Random(seed)
    distractors = []
    for number in range(count):
        base = chunks[number % len(chunks)]
        names: dict[str, str] = {}

        def rename(match: re.Match) -> str:
            return names.setdefault(match.group(0), _name(rng))

        def perturb(text: str) -> str:
            return _NUMBER.sub(lambda match: _digits(match, rng), _LATIN_NAME.sub(rename, text))

        distractors.append(
            base.model_copy(
                update={
                    "chunk_id": f"synthetic_{number:06d}",
                    "doc_id": f"synthetic_{number // 8:05d}",
                    "title": perturb(base.title),
                    "section": perturb(base.section) if base.section else None,
                    "text": perturb(base.text),
                    "source": f"synthetic/{number // 8:05d}.txt",
                    "format": "txt",
                    "page": None,
                }
            )
        )
    return distractors
