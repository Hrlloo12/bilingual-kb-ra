import importlib.util

import pytest

from rag.config import REPO_ROOT
from rag.facts import FactBase
from rag.kb_render import load_templates, render_document, write_document
from rag.normalize import normalize_for_search
from rag.parsers import parse_file

FACT_BASE = FactBase.load(REPO_ROOT / "data" / "facts")
TEMPLATES = load_templates(REPO_ROOT / "data" / "templates")
HAS_WEASYPRINT = importlib.util.find_spec("weasyprint") is not None


def _round_trip_cases():
    for template in TEMPLATES:
        marks = [pytest.mark.skipif(not HAS_WEASYPRINT, reason="weasyprint not installed")] if template.format == "pdf" else []
        yield pytest.param(template, id=template.doc_id, marks=marks)


def _squash(text: str) -> str:
    return " ".join(normalize_for_search(text).split())


@pytest.mark.parametrize("template", list(_round_trip_cases()))
def test_round_trip_preserves_title_headings_and_facts(template, tmp_path):
    rendered = render_document(template, FACT_BASE)
    parsed = parse_file(write_document(rendered, tmp_path))

    assert _squash(parsed.title) == _squash(rendered.title)
    assert [_squash(section.heading or "") for section in parsed.sections] == [
        _squash(section.heading) for section in rendered.sections
    ]
    parsed_text = _squash(parsed.full_text)
    language = "en" if rendered.language == "en" else "ar"
    for section in rendered.sections:
        for fact_id in section.fact_ids:
            fact = FACT_BASE.get(fact_id)
            assert _squash(fact.statement.get(language)) in parsed_text or _squash(fact.value.get(language)) in parsed_text, fact_id


@pytest.mark.skipif(importlib.util.find_spec("fitz") is None, reason="pymupdf not installed")
def test_restore_left_to_right_runs_in_right_to_left_reading():
    from rag.parsers.pdf_parser import restore_left_to_right_runs

    assert restore_left_to_right_runs("السعر 005,1 لايرالرمز 012-RB-HQ") == "السعر 1,500 لايرالرمز QH-BR-210"
    assert restore_left_to_right_runs("في نظام tnemeganaM redrO") == "في نظام Order Management"
    assert restore_left_to_right_runs("بنسبة %51") == "بنسبة 15%"


@pytest.mark.skipif(not HAS_WEASYPRINT, reason="weasyprint not installed")
def test_pdf_page_numbers_follow_page_breaks(tmp_path):
    template = next(template for template in TEMPLATES if template.doc_id == "returns_policy_ar")
    parsed = parse_file(write_document(render_document(template, FACT_BASE), tmp_path))
    assert parsed.page_count == 2
    pages_by_heading = {section.heading: {block.page for block in section.blocks} for section in parsed.sections}
    assert pages_by_heading["الطلبات المتأخرة"] == {2}
    assert pages_by_heading["شروط الاسترجاع"] == {1}
