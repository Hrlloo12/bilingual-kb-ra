from rag.chunking import chunk_document
from rag.langdetect import detect_language
from rag.schemas import DocumentMetadata, ParsedBlock, ParsedDocument, ParsedSection

METADATA = DocumentMetadata(
    doc_id="returns_policy_ar", title="سياسة الاسترجاع", language="ar", domain="returns", format="pdf", source="returns/returns_policy_ar.pdf"
)


def _document(*sections: ParsedSection) -> ParsedDocument:
    return ParsedDocument(title="سياسة الاسترجاع", format="pdf", page_count=2, sections=list(sections))


def test_chunks_never_cross_sections_and_keep_metadata():
    document = _document(
        ParsedSection(heading="شروط الاسترجاع", blocks=[ParsedBlock(text="أ" * 50, page=1), ParsedBlock(text="ب" * 50, page=1)]),
        ParsedSection(heading="الطلبات المتأخرة", blocks=[ParsedBlock(text="ج" * 50, page=2)]),
    )
    chunks = chunk_document(document, METADATA, max_chars=1200)
    assert [chunk.chunk_id for chunk in chunks] == ["returns_policy_ar#1", "returns_policy_ar#2"]
    assert [chunk.section for chunk in chunks] == ["شروط الاسترجاع", "الطلبات المتأخرة"]
    assert [chunk.page for chunk in chunks] == [1, 2]
    assert chunks[0].source == "returns/returns_policy_ar.pdf"


def test_large_sections_split_on_blocks_and_sentences():
    sentence = "هذه جملة اختبارية طويلة. "
    document = _document(ParsedSection(heading="قسم", blocks=[ParsedBlock(text=sentence * 40, page=1)]))
    chunks = chunk_document(document, METADATA, max_chars=300)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 300 for chunk in chunks)


def test_detect_language_buckets():
    assert detect_language("ما هي سياسة الاسترجاع للطلبات المتأخرة؟") == "ar"
    assert detect_language("What is the SLA for premium support tickets?") == "en"
    assert detect_language("أبغى أعرف الـ pricing حق الباقة المؤسسية") == "mixed"
