from collections import Counter

import pytest

from rag.config import REPO_ROOT
from rag.facts import FactBase
from rag.kb_render import (
    check_coverage,
    load_templates,
    localize_digits,
    parse_blocks,
    render_document,
    to_html,
    to_text,
)


@pytest.fixture(scope="module")
def documents():
    fact_base = FactBase.load(REPO_ROOT / "data" / "facts")
    return fact_base, [render_document(template, fact_base) for template in load_templates(REPO_ROOT / "data" / "templates")]


def test_every_fact_rendered_with_declared_coverage(documents):
    fact_base, rendered = documents
    assert check_coverage(rendered, fact_base) == []


def test_corpus_mix(documents):
    _, rendered = documents
    assert len(rendered) == 18
    assert Counter(document.language for document in rendered) == {"ar": 7, "en": 7, "mixed": 4}
    assert set(Counter(document.format for document in rendered)) == {"pdf", "docx", "html", "txt"}
    assert sum(1 for document in rendered if document.digits == "eastern") == 3


def test_eastern_documents_contain_no_western_digits(documents):
    _, rendered = documents
    for document in rendered:
        if document.digits == "eastern":
            text = to_text(document)
            assert not any(character.isascii() and character.isdigit() for character in text)


def test_localize_digits_keeps_product_codes():
    assert localize_digits("QH-BR-210 بسعر 7,800 من 10:00", "eastern") == "QH-BR-210 بسعر ٧,٨٠٠ من ١٠:٠٠"


def test_parse_blocks_detects_bullets_and_paragraphs():
    blocks = parse_blocks("first line\ncontinues\n\n- one\n- two\n")
    assert [block.kind for block in blocks] == ["paragraph", "bullets"]
    assert blocks[0].items == ["first line continues"]
    assert blocks[1].items == ["one", "two"]


def test_html_direction_follows_language(documents):
    _, rendered = documents
    by_id = {document.doc_id: document for document in rendered}
    assert 'dir="rtl"' in to_html(by_id["returns_policy_ar"])
    assert 'dir="ltr"' in to_html(by_id["support_plans_en"])
