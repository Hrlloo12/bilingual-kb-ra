from __future__ import annotations

import re
import unicodedata

_DIGIT_TRANSLATION = str.maketrans(
    {
        **{0x0660 + offset: str(offset) for offset in range(10)},
        **{0x06F0 + offset: str(offset) for offset in range(10)},
        0x066B: ".",
        0x066C: ",",
    }
)
_EASTERN_DIGIT_TRANSLATION = str.maketrans({str(offset): chr(0x0660 + offset) for offset in range(10)})
_LETTER_TRANSLATION = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه"})

_ARABIC_MARKS = re.compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭـ]")
_INVISIBLE = re.compile("[​-‏‪-‮⁦-⁩﻿]")
_HORIZONTAL_SPACE = re.compile(r"[ \t\f\v]+")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
_ARABIC_LETTER = re.compile("[ء-غف-يٱ-ۓ]")
_LATIN_LETTER = re.compile("[A-Za-z]")


def fold_digits(text: str) -> str:
    return text.translate(_DIGIT_TRANSLATION)


def to_eastern_digits(text: str) -> str:
    return text.translate(_EASTERN_DIGIT_TRANSLATION)


def normalize_display(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE.sub("", text)
    lines = (_HORIZONTAL_SPACE.sub(" ", line).strip() for line in text.splitlines())
    return _EXCESS_BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def strip_arabic_marks(text: str) -> str:
    return _ARABIC_MARKS.sub("", text)


def normalize_for_dense(text: str) -> str:
    return strip_arabic_marks(fold_digits(normalize_display(text)))


def normalize_for_search(text: str) -> str:
    return normalize_for_dense(text).translate(_LETTER_TRANSLATION).lower()


def extract_numbers(text: str) -> list[float]:
    return [float(match.replace(",", "")) for match in _NUMBER.findall(fold_digits(text))]


def count_script_letters(text: str) -> tuple[int, int]:
    return len(_ARABIC_LETTER.findall(text)), len(_LATIN_LETTER.findall(text))
