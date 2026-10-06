import pytest

from rag.config import REPO_ROOT
from rag.dataset_builder import (
    assign_splits,
    bucket_for,
    clean_generated,
    embedding_rows,
    fact_components,
    mine_hard_negatives,
)
from rag.facts import FactBase

FACT_BASE = FactBase.load(REPO_ROOT / "data" / "facts")


def test_components_join_facts_sharing_a_section():
    manifest = [
        {"doc_id": "a", "sections": [{"fact_ids": ["x", "y"]}, {"fact_ids": ["z"]}]},
        {"doc_id": "b", "sections": [{"fact_ids": ["y", "w"]}]},
    ]
    assert fact_components(manifest) == [["w", "x", "y"], ["z"]]


def test_splits_keep_components_together():
    components = [["catalog.aria_bed.price"], ["catalog.noor_bed.price", "catalog.noor_bed.dimensions"]]
    splits = assign_splits(components, FACT_BASE, seed=1)
    assert splits["catalog.noor_bed.price"] == splits["catalog.noor_bed.dimensions"]


def test_bucket_assignment_follows_fact_coverage():
    english_only = FACT_BASE.get("installation.kitchen.fee")
    arabic_only = FACT_BASE.get("pricing.support_premium.monthly")
    assert bucket_for("ar_msa", english_only) == ("ar_en", True)
    assert bucket_for("en", arabic_only) == ("en_ar", True)
    assert bucket_for("en", english_only) == ("en_en", False)
    assert bucket_for("ar_dialect", english_only) == ("dialect", True)
    assert bucket_for("mixed", arabic_only) == ("mixed", False)


def test_cleaning_rejects_leaks_wrong_script_and_duplicates():
    splits = {"warranty.kitchen.duration": "train"}
    raw = [
        {
            "fact_id": "warranty.kitchen.duration",
            "parsed": {
                "ar_msa": "ما مدة ضمان المطابخ ضد عيوب التصنيع؟",
                "ar_msa_long": "Kitchen warranty question written in English by mistake for the Arabic field here please",
                "en": "Is the kitchen warranty 2 years long?",
                "en_long": "We installed a new kitchen from Qimam Home last spring and one cabinet door is already loose, so how long does the warranty on kitchens last?",
                "ar_dialect": "ما مدة ضمان المطابخ ضد عيوب التصنيع؟",
                "mixed": "كم مدة الـ warranty على المطابخ؟",
            },
        }
    ]
    queries, report = clean_generated(raw, FACT_BASE, splits)
    assert {query["source"] for query in queries} == {"generated:ar_msa", "generated:en_long", "generated:mixed"}
    assert report["rejected"] == {"wrong_script": 1, "answer_leak": 1, "duplicate": 1}


def test_product_codes_are_not_treated_as_answer_leaks():
    splits = {"catalog.aria_bed.price": "train"}
    raw = [{"fact_id": "catalog.aria_bed.price", "parsed": {"en": "How much does the QH-BR-301 bed cost?"}}]
    queries, _ = clean_generated(raw, FACT_BASE, splits)
    assert [query["query"] for query in queries] == ["How much does the QH-BR-301 bed cost?"]


def test_mining_never_returns_positives_or_held_out_chunks():
    numpy = pytest.importorskip("numpy")
    labels = {
        "c1": frozenset({"warranty.kitchen.duration"}),
        "c2": frozenset({"warranty.kitchen.duration"}),
        "c3": frozenset({"warranty.bedroom.duration"}),
        "c4": frozenset({"hr.probation"}),
        "c5": frozenset(),
    }
    chunk_ids = list(labels)
    query = {"id": "q1", "query": "kitchen warranty", "relevant_fact_ids": ["warranty.kitchen.duration"]}
    mined = mine_hard_negatives(
        [query], chunk_ids, labels, numpy.ones((1, 2)), numpy.eye(5, 2), FACT_BASE, excluded={"c4"}, top_k=10
    )[0]
    negative_ids = [negative["chunk_id"] for negative in mined["negatives"]]
    assert mined["positives"] == ["c1", "c2"]
    assert set(negative_ids) == {"c3", "c5"}
    assert next(negative for negative in mined["negatives"] if negative["chunk_id"] == "c3")["confusable"]

    rows = embedding_rows([query], [mined], {cid: cid for cid in chunk_ids}, {"warranty.kitchen.duration": 0}, 1, 2, 0)
    assert {row["positive"] for row in rows} == {"c1", "c2"}
    assert all(row["negative_1"] == "c3" for row in rows)
