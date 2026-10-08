from rag.config import load_serving_config
from rag.generation import Generation, read_stream
from rag.retrieval.hybrid import Candidate, HybridResult
from rag.schemas import Chunk
from rag.smart_search import SmartSearch


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("#")[0],
        position=0,
        title="Returns policy",
        section="Return window",
        page=1,
        language="en",
        domain="returns",
        format="txt",
        source=f"returns/{chunk_id}.txt",
        text=text,
    )


class StubRetriever:
    def retrieve(self, query, limit=None):
        chunks = [make_chunk("returns_en#1", "Items can be returned within 14 days."), make_chunk("warranty_en#2", "Kitchens carry a 5-year warranty.")]
        return HybridResult(candidates=[Candidate(chunk=chunk, fused_score=0.1) for chunk in chunks], latency_ms={"bm25": 1.0})


class StubReranker:
    def __init__(self, scores):
        self.scores = scores

    def rerank(self, query, candidates):
        for candidate, score in zip(candidates, self.scores):
            candidate.rerank_score = score
        return sorted(candidates, key=lambda candidate: -candidate.rerank_score)


class StubGenerator:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    def generate(self, query, language, chunks):
        self.calls += 1
        return Generation(text=self.text, prompt_tokens=10, completion_tokens=5, latency_ms=2.0, first_token_ms=1.0)


def build(scores, text, threshold=0.5):
    config = load_serving_config()
    config = config.model_copy(update={"smart_search": config.smart_search.model_copy(update={"abstain_threshold": threshold, "context_top_k": 2})})
    search = SmartSearch.__new__(SmartSearch)
    search.settings = config.smart_search
    search.retriever = StubRetriever()
    search.reranker = StubReranker(scores)
    search.generator = StubGenerator(text)
    return search


def test_answer_cites_metadata_of_marked_passage():
    response = build([0.2, 0.9], "Kitchens have a 5-year warranty [1].").search("How long is the kitchen warranty?")
    assert response.status == "answered"
    assert [citation.chunk_id for citation in response.citations] == ["warranty_en#2"]
    assert response.citations[0].source == "returns/warranty_en#2.txt"
    assert response.citations[0].page == 1
    assert response.citations[0].id == 1
    assert response.citations[0].snippet == "Kitchens carry a 5-year warranty."
    assert response.citations[0].relevance == 0.9
    assert response.mode == "smart_ai_search"
    assert {"preprocessing", "retrieval", "rerank", "time_to_first_token", "generation_total", "post_checks", "total"} <= set(response.latency_ms)
    assert response.latency_ms["time_to_first_token"] >= response.latency_ms["generation_first_token"] == 1.0


def test_low_reranker_score_abstains_without_calling_generator():
    search = build([0.1, 0.2], "should not be used")
    response = search.search("Do you rent furniture for events?")
    assert response.status == "not_found"
    assert response.answer == "NOT_FOUND"
    assert response.abstain_reason == "low_retrieval_confidence"
    assert search.generator.calls == 0


def test_unsupported_number_turns_answer_into_not_found():
    response = build([0.9, 0.2], "You can return items within 30 days [1].").search("What is the return window?")
    assert response.status == "not_found"
    assert response.abstain_reason == "unsupported_numbers_or_codes"
    assert response.post_checks["unsupported_numbers"] == [30.0]


def test_missing_markers_fall_back_to_top_passage():
    response = build([0.9, 0.2], "You can return items within 14 days.").search("What is the return window?")
    assert response.status == "answered"
    assert response.post_checks["citation_fallback"]
    assert [citation.chunk_id for citation in response.citations] == ["returns_en#1"]


def test_abstained_answer_has_no_first_token_latency():
    response = build([0.1, 0.2], "unused").search("Do you rent furniture for events?")
    assert "time_to_first_token" not in response.latency_ms and "generation_total" not in response.latency_ms


def test_rank_hook_sees_reranked_candidates_before_generation():
    seen = []
    build([0.2, 0.9], "Kitchens have a 5-year warranty [1].").search(
        "Kitchen warranty?", on_ranked=lambda ranked: seen.extend(candidate.chunk.chunk_id for candidate in ranked)
    )
    assert seen == ["warranty_en#2", "returns_en#1"]


def test_stream_reader_joins_deltas_and_reads_usage():
    lines = [
        'data: {"choices": [{"delta": {"role": "assistant"}}]}',
        "",
        'data: {"choices": [{"delta": {"content": "Kitchens "}}]}',
        'data: {"choices": [{"delta": {"content": "have a 5-year warranty [1]."}}]}',
        'data: {"choices": [], "usage": {"prompt_tokens": 120, "completion_tokens": 9}}',
        "data: [DONE]",
    ]
    text, usage, first_token_ms, total_ms = read_stream(lines)
    assert text == "Kitchens have a 5-year warranty [1]."
    assert usage == {"prompt_tokens": 120, "completion_tokens": 9}
    assert first_token_ms is not None and 0 <= first_token_ms <= total_ms
