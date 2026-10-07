from rag.generation import NOT_FOUND, build_messages
from rag.grounding import ground, parse_markers, strip_markers, unsupported_codes, unsupported_numbers
from rag.schemas import Chunk


def make_chunk(chunk_id: str, text: str, language: str = "en") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("#")[0],
        position=0,
        title="Warranty policy",
        section="Kitchens",
        page=2,
        language=language,
        domain="warranty",
        format="pdf",
        source=f"warranty/{chunk_id}.pdf",
        text=text,
    )


CHUNKS = [
    make_chunk("warranty_en#1", "Kitchens carry a 5-year warranty. Model QK-220 costs 1,450 SAR."),
    make_chunk("warranty_ar#1", "ضمان المطابخ خمس سنوات ورسوم التركيب ٣٠٠ ريال.", "ar"),
]


def test_markers_accept_arabic_digits_lists_and_reject_out_of_range():
    cited, invalid = parse_markers("The warranty is 5 years [1][٢]. Fee [1, 2] and [7].", 2)
    assert cited == [1, 2]
    assert invalid == [7]
    assert strip_markers("It is 5 years [1].") == "It is 5 years."


def test_numbers_must_appear_in_evidence_after_digit_folding():
    evidence = [chunk.text for chunk in CHUNKS]
    assert unsupported_numbers("The fee is 300 SAR and the price is 1450 SAR.", evidence) == []
    assert unsupported_numbers("The fee is 350 SAR.", evidence) == [350.0]
    assert unsupported_codes("Model QK-220 and QK-999.", evidence) == ["QK-999"]


def test_ground_uses_only_cited_passages_as_evidence():
    result = ground("The installation fee is 300 SAR [1].", "What is the kitchen installation fee?", "en", CHUNKS)
    assert result.cited == [1]
    assert result.checks["unsupported_numbers"] == [300.0]
    result = ground("The installation fee is 300 SAR [2].", "What is the kitchen installation fee?", "en", CHUNKS)
    assert result.checks["unsupported_numbers"] == []
    assert result.checks["language_ok"]


def test_numbers_in_the_question_are_allowed():
    result = ground("No, the warranty is 5 years, not 3 [1].", "Is the kitchen warranty 3 years?", "en", CHUNKS)
    assert result.checks["unsupported_numbers"] == []


def test_not_found_and_language_mismatch():
    assert ground("NOT_FOUND", "q", "en", CHUNKS).not_found
    assert ground("not found", "q", "en", CHUNKS).not_found
    result = ground("مدة الضمان خمس سنوات [2].", "How long is the warranty?", "en", CHUNKS)
    assert not result.checks["language_ok"]


def test_prompt_numbers_passages_in_context_order():
    messages = build_messages("كم مدة الضمان؟", "ar", CHUNKS)
    assert "Modern Standard Arabic" in messages[0]["content"]
    assert NOT_FOUND in messages[0]["content"]
    assert messages[1]["content"].index("[1] Warranty policy | Kitchens") < messages[1]["content"].index("[2] Warranty policy")


def test_number_set_ignores_product_codes_and_citation_markers():
    from rag.evaluation.answers import number_set

    assert number_set("أبعاد ErgoPro X3 (الرمز QH-OC-023) هي 68 × 66 × 125 سم [1].") == [3.0, 66.0, 68.0, 125.0]
    assert number_set("The Nexa Pro (code QH-DK-702) is 2,980 SAR [2].") == [2980.0]
