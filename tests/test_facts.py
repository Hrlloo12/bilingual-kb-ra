import pytest
from pydantic import ValidationError

from rag.config import REPO_ROOT
from rag.facts import Fact, FactBase

FACTS_DIR = REPO_ROOT / "data" / "facts"


@pytest.fixture(scope="module")
def fact_base() -> FactBase:
    return FactBase.load(FACTS_DIR)


def test_domains_loaded(fact_base):
    assert {fact_file.domain for fact_file in fact_base.files} >= {
        "support",
        "pricing",
        "warranty",
        "returns",
        "branches_delivery",
        "it_access",
        "catalog",
        "stores",
        "hr",
        "expenses",
        "security",
        "payments",
    }
    assert 250 <= len(fact_base.facts) <= 400


def test_entity_tables_expand_with_group_confusables(fact_base):
    price = fact_base.get("catalog.aria_bed.price")
    assert price.numbers == (3950.0,)
    assert "QH-BR-301" in price.statement.en and "QH-BR-301" in price.statement.ar
    assert "catalog.noor_bed.price" in fact_base.confusables("catalog.aria_bed.price")
    assert "catalog.zaha_island.price" not in fact_base.confusables("catalog.aria_bed.price")


def test_both_exclusive_languages_present(fact_base):
    exclusives = [fact.exclusive_language for fact in fact_base.facts.values()]
    assert exclusives.count("ar") >= 10
    assert exclusives.count("en") >= 10


def test_confusables_are_symmetric(fact_base):
    for fact_id in fact_base.facts:
        for other_id in fact_base.confusables(fact_id):
            assert fact_id in fact_base.confusables(other_id)


def test_hard_negative_pairs_from_brief_exist(fact_base):
    assert "support.standard.first_response" in fact_base.confusables("support.premium.resolution")
    assert "warranty.kitchen.duration" in fact_base.confusables("warranty.bedroom.duration")


def _fact(**overrides):
    base = {
        "id": "test.sample.value",
        "subject": {"ar": "س", "en": "s"},
        "value": {"ar": "5 أيام", "en": "5 days"},
        "statement": {"ar": "خلال 5 أيام.", "en": "Within 5 days."},
        "numbers": [5],
        "coverage": ["en"],
    }
    base.update(overrides)
    return Fact.model_validate(base)


def test_declared_number_must_appear_in_english_statement():
    with pytest.raises(ValidationError):
        _fact(numbers=[7])


def test_arabic_number_must_match_english_statement():
    with pytest.raises(ValidationError):
        _fact(statement={"ar": "خلال ٦ أيام.", "en": "Within 5 days."})


def test_eastern_digits_accepted_in_arabic_statement():
    assert _fact(statement={"ar": "خلال ٥ أيام.", "en": "Within 5 days."}).numbers == (5.0,)
