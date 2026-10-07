import pytest

from rag.evaluation.generation import fact_coverage, parse_verdict, precision, rouge_l, tokens


def test_rouge_l_handles_arabic_and_ignores_citation_markers():
    reference = "سعر سرير Noor هو 4,600 ريال سعودي."
    assert rouge_l("سعر سرير Noor هو 4,600 ريال سعودي [1].", reference) == pytest.approx(1.0)
    assert 0 < rouge_l("سعر سرير Aria هو 3,950 ريال سعودي [1].", reference) < 1
    assert rouge_l("", reference) == 0.0


def test_tokens_keep_arabic_letters():
    assert tokens("أبعاد السرير") == ["ابعاد", "السرير"]


def test_fact_level_coverage_and_precision():
    labels = {"a#1": frozenset({"f1"}), "a_en#1": frozenset({"f1"}), "b#1": frozenset({"f2"})}
    assert fact_coverage(["a#1", "a_en#1"], labels, {"f1"}) == 1.0
    assert fact_coverage(["b#1"], labels, {"f1", "f2"}) == 0.5
    assert precision(["a#1", "b#1"], labels, {"f1"}) == 0.5
    assert precision([], labels, {"f1"}) is None


def test_parse_verdict_accepts_wrapped_json_and_rejects_malformed():
    assert parse_verdict('Sure: {"supported": true, "unsupported_claims": [], "relevance": "full"}') == {
        "supported": True,
        "relevance": "full",
        "unsupported_claims": [],
    }
    assert parse_verdict('{"supported": "yes", "relevance": "full"}') is None
    assert parse_verdict('{"supported": false, "relevance": "maybe"}') is None
    assert parse_verdict("no json here") is None
