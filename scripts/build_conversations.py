from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from rag.config import REPO_ROOT
from rag.facts import FactBase

DATA_DIR = REPO_ROOT / "data"
LANGUAGE_PAIRS = (("en", "en"), ("ar", "ar"), ("en", "ar"), ("ar", "mixed"))
BARE_PAIRS = (("en", "en"), ("ar", "ar"))
FIRST_TURN_SOURCES = {"en": ("generated:en", "generated:en_long"), "ar": ("generated:ar_msa", "generated:mixed->ar_msa", "generated:ar_msa_long")}
GOLD_SOURCES = {"en": ("generated:en",), "ar": ("generated:ar_msa", "generated:mixed->ar_msa"), "mixed": ("generated:mixed", "generated:ar_msa")}
STANDALONE_EVERY = 4

FOLLOW_UPS = {
    ("catalog", "price"): {"en": "And how much does it cost?", "ar": "وكم سعره؟", "mixed": "وكم الـ price حقه؟"},
    ("catalog", "dimensions"): {"en": "What are its dimensions?", "ar": "وما أبعاده؟", "mixed": "وش الـ dimensions حقه؟"},
    ("installation_services", "duration"): {"en": "And how long does it take to install?", "ar": "وكم يستغرق تركيبه؟", "mixed": "وكم ياخذ وقت الـ installation؟"},
    ("installation_services", "technicians"): {"en": "How many technicians does that need?", "ar": "وكم فنيا يحتاج تركيبه؟", "mixed": "وكم technician يحتاج؟"},
    ("stores", "location"): {"en": "Where exactly is it?", "ar": "وأين يقع؟", "mixed": "وين الـ location حقه؟"},
    ("stores", "phone"): {"en": "What's its phone number?", "ar": "وما رقم هاتفه؟", "mixed": "وش رقم الـ phone حقه؟"},
    ("stores", "service"): {"en": "What special service does it offer?", "ar": "وما الخدمة المميزة التي يقدمها؟", "mixed": "وش الـ service اللي يقدمها؟"},
    ("travel", "per_diem"): {"en": "And what is their daily allowance?", "ar": "وكم بدلهم اليومي؟", "mixed": "وكم الـ per diem حقهم؟"},
    ("travel", "hotel_cap"): {"en": "What is their hotel limit?", "ar": "وما سقف الفندق لهم؟", "mixed": "وكم الـ hotel cap حقهم؟"},
}
BARE = {
    "price": {"en": "Price?", "ar": "السعر؟"},
    "dimensions": {"en": "Dimensions?", "ar": "الأبعاد؟"},
    "duration": {"en": "Installation time?", "ar": "مدة التركيب؟"},
    "technicians": {"en": "Number of technicians?", "ar": "عدد الفنيين؟"},
    "location": {"en": "Address?", "ar": "العنوان؟"},
    "phone": {"en": "Phone number?", "ar": "رقم الهاتف؟"},
    "service": {"en": "Special service?", "ar": "الخدمة المميزة؟"},
    "per_diem": {"en": "Daily allowance?", "ar": "البدل اليومي؟"},
    "hotel_cap": {"en": "Hotel cap?", "ar": "سقف الفندق؟"},
}


def bare_english(subject: str) -> str:
    for article in ("a ", "an ", "the "):
        if subject.lower().startswith(article):
            return subject[len(article):]
    return subject


def entity_switch(domain: str, subject_en: str, subject_ar: str, language: str) -> str:
    name = bare_english(subject_en)
    if domain == "stores":
        return {"en": f"What about the {name} showroom?", "ar": f"وماذا عن معرض {subject_ar}؟", "mixed": f"طيب و showroom {name}؟"}[language]
    if domain == "catalog":
        return {"en": f"What about the {name}?", "ar": f"وماذا عن {subject_ar}؟", "mixed": f"طيب والـ {name}؟"}[language]
    return {"en": f"What about {subject_en}?", "ar": f"وماذا عن {subject_ar}؟", "mixed": f"طيب وبالنسبة لـ {name}؟"}[language]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pick(queries: list[dict], fact_id: str, sources: tuple[str, ...]) -> dict | None:
    for source in sources:
        for row in queries:
            if row["relevant_fact_ids"] == [fact_id] and row["source"] == source:
                return row
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build multi-turn conversations whose second turn targets a held-out fact.")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "interactive")
    args = parser.parse_args(argv)

    fact_base = FactBase.load(DATA_DIR / "facts")
    splits = json.loads((DATA_DIR / "splits" / "fact_splits.json").read_text(encoding="utf-8"))["splits"]
    queries = [row for split in ("train", "validation", "test") for row in read_jsonl(DATA_DIR / f"{split}.jsonl")]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("validation", "test"):
        conversations = []

        def add(kind: str, first: dict, second_query: str, second_language: str, target_id: str, pair: tuple[str, str]) -> None:
            gold = pick([row for row in queries if row["split"] == split], target_id, GOLD_SOURCES[second_language])
            fact = fact_base.get(target_id)
            conversations.append(
                {
                    "id": f"{split}_conv_{len(conversations) + 1:04d}",
                    "split": split,
                    "type": kind,
                    "languages": f"{pair[0]}>{pair[1]}",
                    "turns": [
                        {"query": first["query"], "language": first["language"], "relevant_fact_ids": first["relevant_fact_ids"]},
                        {
                            "query": second_query,
                            "language": second_language,
                            "needs_rewrite": kind != "standalone",
                            "relevant_fact_ids": [target_id],
                            "reference_answer": fact.statement.get("en" if second_language == "en" else "ar"),
                            "gold_standalone": gold["query"] if gold else None,
                        },
                    ],
                }
            )

        for fact_file in fact_base.entities.values():
            domain = fact_file.domain
            entities = sorted(fact_file.entities, key=lambda entity: entity.key)
            for entity in entities:
                for attribute in sorted(entity.coverage):
                    target_id = fact_file.entity_fact_id(entity.key, attribute)
                    if splits.get(target_id) != split:
                        continue
                    for other in sorted(entity.coverage):
                        if other == attribute:
                            continue
                        context_id = fact_file.entity_fact_id(entity.key, other)
                        for pair in LANGUAGE_PAIRS:
                            first = pick(queries, context_id, FIRST_TURN_SOURCES[pair[0]])
                            if first:
                                add("attribute_follow_up", first, FOLLOW_UPS[(domain, attribute)][pair[1]], pair[1], target_id, pair)
                        for pair in BARE_PAIRS:
                            first = pick(queries, context_id, FIRST_TURN_SOURCES[pair[0]])
                            if first:
                                add("bare_follow_up", first, BARE[attribute][pair[1]], pair[1], target_id, pair)
                    peer = next(
                        (candidate for candidate in entities if candidate.key != entity.key and candidate.group == entity.group and attribute in candidate.coverage),
                        None,
                    )
                    if peer:
                        context_id = fact_file.entity_fact_id(peer.key, attribute)
                        for pair in LANGUAGE_PAIRS:
                            first = pick(queries, context_id, FIRST_TURN_SOURCES[pair[0]])
                            if first:
                                add("entity_switch", first, entity_switch(domain, entity.subject.en, entity.subject.ar, pair[1]), pair[1], target_id, pair)

        follow_ups = list(conversations)
        held_out = [row for row in read_jsonl(DATA_DIR / f"{split}.jsonl") if row["relevant_fact_ids"]]
        for index, row in enumerate(held_out[::STANDALONE_EVERY]):
            context = follow_ups[index % len(follow_ups)]["turns"][0]
            first = {"query": context["query"], "language": context["language"], "relevant_fact_ids": context["relevant_fact_ids"]}
            add("standalone", first, row["query"], row["language"], row["relevant_fact_ids"][0], (context["language"], row["language"]))
            conversations[-1]["turns"][1]["gold_standalone"] = row["query"]
            conversations[-1]["turns"][1]["relevant_fact_ids"] = row["relevant_fact_ids"]

        path = args.output_dir / f"conversations_{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in conversations), encoding="utf-8")
        print(split, len(conversations), dict(Counter(row["type"] for row in conversations)), dict(Counter(row["languages"] for row in conversations)), "missing gold:", sum(1 for row in conversations if not row["turns"][1]["gold_standalone"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
