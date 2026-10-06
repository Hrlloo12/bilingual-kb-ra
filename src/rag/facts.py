from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag.normalize import extract_numbers

Coverage = Literal["ar", "en", "mixed"]


class Bilingual(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ar: str = Field(min_length=1)
    en: str = Field(min_length=1)

    def get(self, language: str) -> str:
        return self.en if language == "en" else self.ar


class Fact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)+$")
    subject: Bilingual
    value: Bilingual
    statement: Bilingual
    numbers: tuple[float, ...] = ()
    coverage: tuple[Coverage, ...] = Field(min_length=1)
    confusable_with: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check_consistency(self) -> Fact:
        if len(set(self.coverage)) != len(self.coverage):
            raise ValueError(f"{self.id}: duplicate coverage entries")
        if self.id in self.confusable_with:
            raise ValueError(f"{self.id}: fact cannot be confusable with itself")
        english_numbers = set(extract_numbers(self.statement.en))
        missing = [number for number in self.numbers if number not in english_numbers]
        if missing:
            raise ValueError(f"{self.id}: declared numbers {missing} missing from English statement")
        for language in ("ar", "en"):
            for field_name in ("statement", "value"):
                text = getattr(self, field_name).get(language)
                unexpected = sorted(set(extract_numbers(text)) - english_numbers)
                if unexpected:
                    raise ValueError(f"{self.id}: {language} {field_name} has numbers {unexpected} not in English statement")
        return self

    @property
    def exclusive_language(self) -> str | None:
        languages = {"en" if entry == "en" else "ar" for entry in self.coverage}
        return languages.pop() if len(languages) == 1 else None


class FactFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    domain: str = Field(pattern=r"^[a-z][a-z_]*$")
    description: str = Field(min_length=1)
    facts: tuple[Fact, ...] = Field(min_length=1)


class FactBase:
    def __init__(self, files: list[FactFile]) -> None:
        self.files = files
        self.facts: dict[str, Fact] = {}
        self.domain_of: dict[str, str] = {}
        for fact_file in files:
            for fact in fact_file.facts:
                if fact.id in self.facts:
                    raise ValueError(f"duplicate fact id {fact.id}")
                self.facts[fact.id] = fact
                self.domain_of[fact.id] = fact_file.domain
        self._confusables = self._build_confusables()

    @classmethod
    def load(cls, directory: Path) -> FactBase:
        paths = sorted(directory.glob("*.yaml"))
        if not paths:
            raise FileNotFoundError(f"no fact files found in {directory}")
        files = [FactFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8"))) for path in paths]
        domains = Counter(fact_file.domain for fact_file in files)
        duplicated = [domain for domain, count in domains.items() if count > 1]
        if duplicated:
            raise ValueError(f"duplicate domains {duplicated}")
        return cls(files)

    def _build_confusables(self) -> dict[str, frozenset[str]]:
        pairs: dict[str, set[str]] = {fact_id: set() for fact_id in self.facts}
        for fact in self.facts.values():
            for other_id in fact.confusable_with:
                if other_id not in self.facts:
                    raise ValueError(f"{fact.id}: unknown confusable fact {other_id}")
                pairs[fact.id].add(other_id)
                pairs[other_id].add(fact.id)
        return {fact_id: frozenset(others) for fact_id, others in pairs.items()}

    def get(self, fact_id: str) -> Fact:
        if fact_id not in self.facts:
            raise KeyError(f"unknown fact {fact_id}")
        return self.facts[fact_id]

    def confusables(self, fact_id: str) -> frozenset[str]:
        return self._confusables[fact_id]

    def summary(self) -> dict[str, dict[str, int]]:
        report: dict[str, dict[str, int]] = {}
        for fact_file in self.files:
            exclusive = Counter(fact.exclusive_language or "both" for fact in fact_file.facts)
            report[fact_file.domain] = {
                "facts": len(fact_file.facts),
                "ar_only": exclusive["ar"],
                "en_only": exclusive["en"],
                "both": exclusive["both"],
                "with_numbers": sum(1 for fact in fact_file.facts if fact.numbers),
                "with_confusables": sum(1 for fact in fact_file.facts if self._confusables[fact.id]),
            }
        return report
