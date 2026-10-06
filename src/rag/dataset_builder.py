from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from rag.facts import Fact, FactBase
from rag.langdetect import arabic_ratio
from rag.normalize import extract_numbers, search_key

SPLITS = ("train", "validation", "test")
SPLIT_RATIOS = {"train": 0.75, "validation": 0.10, "test": 0.15}
QUERY_FIELDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "ar_msa": ("ar", ()),
    "ar_msa_long": ("ar", ("long",)),
    "en": ("en", ()),
    "en_long": ("en", ("long",)),
    "ar_dialect": ("ar", ("dialect",)),
    "mixed": ("mixed", ("code_switch",)),
}
WORD_LIMITS = {"short": (3, 32), "long": (12, 90)}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def fact_components(manifest: list[dict]) -> list[list[str]]:
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for record in manifest:
        for section in record["sections"]:
            facts = section["fact_ids"]
            for fact_id in facts:
                find(fact_id)
            for other in facts[1:]:
                parent[find(other)] = find(facts[0])
    groups: dict[str, list[str]] = defaultdict(list)
    for fact_id in parent:
        groups[find(fact_id)].append(fact_id)
    return sorted((sorted(group) for group in groups.values()), key=lambda group: group[0])


def assign_splits(components: list[list[str]], fact_base: FactBase, seed: int) -> dict[str, str]:
    by_domain: dict[str, list[list[str]]] = defaultdict(list)
    for component in components:
        by_domain[fact_base.domain_of[component[0]]].append(component)
    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    for domain in sorted(by_domain):
        domain_components = by_domain[domain]
        rng.shuffle(domain_components)
        total = sum(len(component) for component in domain_components)
        counts = Counter()
        for component in domain_components:
            split = max(SPLITS, key=lambda name: SPLIT_RATIOS[name] * total - counts[name])
            counts[split] += len(component)
            for fact_id in component:
                assignment[fact_id] = split
    return assignment


def _has_arabic_side(fact: Fact) -> bool:
    return any(entry in ("ar", "mixed") for entry in fact.coverage)


def name_style(fact: Fact) -> str | None:
    if not any("a" <= character.lower() <= "z" for character in fact.subject.ar):
        return None
    digest = int(hashlib.md5(fact.id.encode("utf-8")).hexdigest(), 16)
    return "latin" if digest % 2 == 0 else "arabic_script"


def generation_records(fact_base: FactBase, splits: dict[str, str]) -> list[dict]:
    records = []
    for fact_id in sorted(splits):
        fact = fact_base.get(fact_id)
        records.append(
            {
                "fact_id": fact_id,
                "split": splits[fact_id],
                "subject_en": fact.subject.en,
                "subject_ar": fact.subject.ar,
                "statement_en": fact.statement.en,
                "statement_ar": fact.statement.ar,
                "name_style": name_style(fact),
            }
        )
    return records


def bucket_for(field: str, fact: Fact) -> tuple[str, bool]:
    language = QUERY_FIELDS[field][0]
    if language == "en":
        cross = "en" not in fact.coverage
        return ("en_ar" if cross else "en_en"), cross
    cross = not _has_arabic_side(fact)
    if field == "ar_dialect":
        return "dialect", cross
    if field == "mixed":
        return "mixed", cross
    return ("ar_en" if cross else "ar_ar"), cross


def _script_ok(field: str, text: str) -> bool:
    ratio = arabic_ratio(text) or 0.0
    has_latin = any("a" <= character.lower() <= "z" for character in text)
    language = QUERY_FIELDS[field][0]
    if language == "en":
        return ratio == 0.0
    if language == "mixed":
        return has_latin and 0.3 <= ratio <= 0.97
    return ratio >= 0.5


def _leaks_answer(text: str, fact: Fact) -> bool:
    allowed = set(extract_numbers(f"{fact.subject.en} {fact.subject.ar}"))
    for token in fact.statement.en.split():
        if "-" in token and any(character.isdigit() for character in token) and any(character.isupper() for character in token):
            allowed.update(extract_numbers(token))
    protected = set(fact.numbers) - allowed
    return bool(protected & set(extract_numbers(text)))


def _length_ok(field: str, text: str) -> bool:
    low, high = WORD_LIMITS["long" if field.endswith("_long") else "short"]
    return low <= len(text.split()) <= high


def clean_generated(raw: list[dict], fact_base: FactBase, splits: dict[str, str]) -> tuple[list[dict], dict]:
    rejected: Counter = Counter()
    seen: set[str] = set()
    queries: list[dict] = []
    for record in raw:
        fact_id = record["fact_id"]
        if fact_id not in splits:
            rejected["unknown_fact"] += 1
            continue
        parsed = record.get("parsed") or {}
        fact = fact_base.get(fact_id)
        for field, (language, field_tags) in QUERY_FIELDS.items():
            text = " ".join(str(parsed.get(field, "")).split())
            if not text:
                rejected["empty"] += 1
                continue
            if not _script_ok(field, text):
                rejected["wrong_script"] += 1
                continue
            if not _length_ok(field, text):
                rejected["length"] += 1
                continue
            if _leaks_answer(text, fact):
                rejected["answer_leak"] += 1
                continue
            key = search_key(text)
            if key in seen:
                rejected["duplicate"] += 1
                continue
            seen.add(key)
            bucket, cross = bucket_for(field, fact)
            tags = list(field_tags)
            if cross:
                tags.append("cross_lingual")
            if fact.numbers:
                tags.append("numeric")
            if fact_base.confusables(fact_id):
                tags.append("hard_negative")
            queries.append(
                {
                    "query": text,
                    "language": language,
                    "bucket": bucket,
                    "relevant_fact_ids": [fact_id],
                    "tags": tags,
                    "split": splits[fact_id],
                    "source": f"generated:{field}",
                    "reference_answer": fact.statement.en if language == "en" else fact.statement.ar,
                }
            )
    report = {"accepted": len(queries), "rejected": dict(rejected)}
    return queries, report


def held_out_chunk_ids(labels: dict[str, frozenset[str]], splits: dict[str, str]) -> set[str]:
    return {chunk_id for chunk_id, facts in labels.items() if any(splits.get(fact_id) != "train" for fact_id in facts)}


def mine_hard_negatives(
    queries: list[dict],
    chunk_ids: list[str],
    labels: dict[str, frozenset[str]],
    query_vectors,
    chunk_vectors,
    fact_base: FactBase,
    excluded: set[str],
    top_k: int,
) -> list[dict]:
    import numpy as np

    scores = np.asarray(query_vectors) @ np.asarray(chunk_vectors).T
    records = []
    for row, query in enumerate(queries):
        targets = set(query["relevant_fact_ids"])
        confusable_facts = set().union(*(fact_base.confusables(fact_id) for fact_id in targets)) - targets
        positives = sorted(chunk_id for chunk_id, facts in labels.items() if facts & targets)
        blocked = set(positives) | excluded
        negatives = []
        for column in np.argsort(-scores[row]):
            chunk_id = chunk_ids[column]
            if chunk_id in blocked or labels[chunk_id] & targets:
                continue
            negatives.append(
                {"chunk_id": chunk_id, "score": round(float(scores[row, column]), 4), "confusable": bool(labels[chunk_id] & confusable_facts)}
            )
            if len(negatives) == top_k:
                break
        listed = {negative["chunk_id"] for negative in negatives}
        for column, chunk_id in enumerate(chunk_ids):
            if chunk_id not in blocked and chunk_id not in listed and labels[chunk_id] & confusable_facts:
                negatives.append({"chunk_id": chunk_id, "score": round(float(scores[row, column]), 4), "confusable": True})
        records.append({"query_id": query["id"], "positives": positives, "negatives": negatives})
    return records


def embedding_rows(
    queries: list[dict],
    mined: list[dict],
    passages: dict[str, str],
    component_of: dict[str, int],
    negatives_per_row: int,
    max_positives: int,
    seed: int,
) -> list[dict]:
    from rag.normalize import normalize_for_dense

    rng = random.Random(seed)
    by_id = {record["query_id"]: record for record in mined}
    rows = []
    for query in queries:
        record = by_id[query["id"]]
        positives = record["positives"]
        if len(positives) > max_positives:
            positives = rng.sample(positives, max_positives)
        ranked = sorted(record["negatives"], key=lambda negative: (not negative["confusable"], -negative["score"]))
        chosen = [negative["chunk_id"] for negative in ranked[:negatives_per_row]]
        if len(chosen) < negatives_per_row:
            continue
        for positive in positives:
            row = {
                "anchor": normalize_for_dense(query["query"]),
                "positive": passages[positive],
                "group": component_of[query["relevant_fact_ids"][0]],
                "query_id": query["id"],
            }
            for number, chunk_id in enumerate(chosen, start=1):
                row[f"negative_{number}"] = passages[chunk_id]
            rows.append(row)
    return rows


def assign_ids(queries: list[dict]) -> dict[str, list[dict]]:
    by_split: dict[str, list[dict]] = {split: [] for split in SPLITS}
    for query in sorted(queries, key=lambda item: (item["split"], item["relevant_fact_ids"], item["source"], item["query"])):
        by_split[query["split"]].append(query)
    for split, rows in by_split.items():
        for number, row in enumerate(rows, start=1):
            row["id"] = f"{split}_{number:05d}"
    return by_split
