from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

from rag.evaluation.metrics import latency_summary
from rag.grounding import strip_markers
from rag.normalize import extract_numbers

PRODUCT_CODE = re.compile(r"\b[A-Za-z]{2,}(?:-[A-Za-z0-9]+)*-\d+[A-Za-z0-9]*\b")


def number_set(text: str) -> list[float]:
    return sorted(set(extract_numbers(PRODUCT_CODE.sub(" ", strip_markers(text)))))


def rate(values: list[bool]) -> float | None:
    return round(float(np.mean(values)), 4) if values else None


def summarize(records: list[dict]) -> dict:
    answerable = [record for record in records if record["answerable"]]
    unanswerable = [record for record in records if not record["answerable"]]
    answered = [record for record in answerable if record["status"] == "answered"]
    cited = [record for record in answered if record["cited_chunk_ids"]]
    with_numbers = [record for record in answered if record["reference_numbers"]]
    numeric_answers = [record for record in answered if record["answer_numbers"] and record["reference_numbers"]]
    summary = {
        "queries": len(records),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "answerable_answered_rate": rate([record["status"] == "answered" for record in answerable]),
        "answerable_false_not_found_rate": rate([record["status"] == "not_found" for record in answerable]),
        "unanswerable_not_found_rate": rate([record["status"] == "not_found" for record in unanswerable]),
        "context_hit_rate": rate([record["context_hit"] for record in answerable]),
        "citation_hit_rate": rate([any(chunk_id in record["relevant_chunk_ids"] for chunk_id in record["cited_chunk_ids"]) for record in cited]),
        "citation_precision": rate([chunk_id in record["relevant_chunk_ids"] for record in cited for chunk_id in record["cited_chunk_ids"]]),
        "citation_fallback_rate": rate([bool(record["post_checks"].get("citation_fallback")) for record in answered]),
        "language_ok_rate": rate([bool(record["post_checks"].get("language_ok")) for record in answered]),
        "reference_numbers_all_in_answer_rate": rate([set(record["reference_numbers"]) <= set(record["answer_numbers"]) for record in with_numbers]),
        "reference_numbers_n": len(with_numbers),
        "answer_numbers_all_in_reference_rate": rate([set(record["answer_numbers"]) <= set(record["reference_numbers"]) for record in numeric_answers]),
        "answer_numbers_n": len(numeric_answers),
        "abstain_reasons": dict(sorted({reason: sum(1 for record in records if record["abstain_reason"] == reason) for reason in {record["abstain_reason"] for record in records} if reason}.items())),
    }
    stages = sorted({stage for record in records for stage in record["latency_ms"]})
    summary["latency_ms"] = {stage: latency_summary([record["latency_ms"][stage] for record in records if stage in record["latency_ms"]]) for stage in stages}
    summary["latency_ms_answered_total"] = latency_summary([record["latency_ms"]["total"] for record in records if record["status"] == "answered"]) if any(record["status"] == "answered" for record in records) else {}
    return summary


def summarize_by_bucket(records: list[dict]) -> dict:
    by_bucket = defaultdict(list)
    for record in records:
        by_bucket[record["bucket"]].append(record)
    return {"overall": summarize(records), "by_bucket": {bucket: summarize(rows) for bucket, rows in sorted(by_bucket.items())}}
