import pytest

from rag.config import load_serving_config
from rag.followups import FollowupSuggester, parse_followups
from rag.interactive import InteractiveSearch
from rag.rewrite import QueryRewriter, build_rewrite_messages, clean_rewrite, gate
from rag.schemas import Chunk, SmartSearchResponse
from rag.sessions import SessionStore, Turn, valid_session_id

CONFIG = load_serving_config()


class FakeValkey:
    def __init__(self):
        self.lists, self.expiry = {}, {}

    def lrange(self, key, start, end):
        return list(self.lists.get(key, []))

    def rpush(self, key, value):
        self.lists.setdefault(key, []).append(value)

    def ltrim(self, key, start, end):
        self.lists[key] = self.lists.get(key, [])[start:]

    def expire(self, key, seconds):
        self.expiry[key] = seconds

    def ttl(self, key):
        return self.expiry.get(key, -2)

    def delete(self, key):
        return int(self.lists.pop(key, None) is not None)

    def ping(self):
        return True

    def pipeline(self):
        return self

    def execute(self):
        return []


def turn(query="What is the price of the Nexa Pro desk?", status="answered", answer="The Nexa Pro desk costs 2,980 SAR [1]."):
    return Turn(query=query, standalone_query=query, language="en", status=status, answer=answer)


@pytest.mark.parametrize(
    "query",
    [
        "And its dimensions?",
        "What about the Rimal bed?",
        "How much does it cost?",
        "وكم سعره؟",
        "وماذا عن سرير Rimal؟",
        "طيب والـ warranty؟",
        "وش رقم الـ phone حقه؟",
        "ما أبعاده؟",
        "Price?",
        "وش الـ service اللي يقدمها؟",
        "What special service does it offer?",
    ],
)
def test_gate_rewrites_follow_ups(query):
    assert gate(query, [turn()]).rewrite


@pytest.mark.parametrize(
    "query",
    [
        "What is the return window for online orders?",
        "وش أوقات دوام معرض الرياض يوم الجمعة؟",
        "كم مدة ضمان المطابخ ضد عيوب التصنيع؟",
        "كم الـ price حق مكتب Nexa Pro؟",
        "How many days of annual leave do new employees get?",
        "ما هي ميزانية التدريب السنوية لكل موظف في الشركة؟",
        "أنا أبحث عن سرير Rimal برمز QH-BR-303، ما سعره؟",
        "I need to contact the Riyadh Al Rawdah showroom, what is the phone number I can use to reach them?",
        "أحتاج معرفة المبلغ اليومي الذي يحصل عليه الموظفون في الدرجات G1–G3 أثناء رحلات دول الخليج، هل يمكنك توضيح ذلك؟",
    ],
)
def test_gate_skips_standalone_questions(query):
    assert not gate(query, [turn()]).rewrite


def test_gate_never_rewrites_the_first_turn():
    decision = gate("And its dimensions?", [])
    assert not decision.rewrite
    assert decision.reason == "no_history"


def test_rewrite_prompt_includes_previous_answer_only_when_answered():
    messages = build_rewrite_messages("And its size?", turn(), 400)
    assert "Previous answer: The Nexa Pro desk costs 2,980 SAR." in messages[1]["content"]
    messages = build_rewrite_messages("And its size?", turn(status="not_found", answer="NOT_FOUND"), 400)
    assert "Previous answer" not in messages[1]["content"]


def test_clean_rewrite_strips_labels_and_quotes():
    assert clean_rewrite('Standalone question: "What are the dimensions of the Nexa Pro desk?"\nextra') == "What are the dimensions of the Nexa Pro desk?"


def test_session_store_trims_and_expires():
    client = FakeValkey()
    store = SessionStore(client, ttl_s=60, max_turns=2)
    session_id = "abcdef1234567890"
    for number in range(3):
        store.append(session_id, turn(query=f"q{number}"))
    assert [item.query for item in store.history(session_id)] == ["q1", "q2"]
    assert store.ttl(session_id) == 60
    assert store.clear(session_id)
    assert store.history(session_id) == []
    with pytest.raises(ValueError):
        store.history("bad id!")
    assert valid_session_id("abcdef1234567890") and not valid_session_id("x")


class StubSmart:
    def __init__(self):
        self.calls = []

    def search(self, query, language=None, on_ranked=None):
        self.calls.append((query, language))
        if on_ranked is not None:
            on_ranked([])
        return SmartSearchResponse(
            query=query,
            language_detected=language or "en",
            status="answered",
            answer="answer [1]",
            citations=[],
            retrieved=[],
            latency_ms={"retrieval": 2.0, "rerank": 1.0, "time_to_first_token": 3.0, "generation_total": 1.5, "total": 5.0},
        )


class StubSuggester(FollowupSuggester):
    def __init__(self):
        super().__init__(CONFIG.generation, CONFIG.interactive)

    def suggest(self, query, language, chunks):
        return ["ما مدة ضمان مكتب Nexa Pro؟"] if language != "en" else ["What is the warranty on the Nexa Pro desk?"]


class StubRewriter(QueryRewriter):
    def __init__(self):
        super().__init__(CONFIG.generation, CONFIG.interactive)
        self.prompts = []

    def complete(self, messages):
        self.prompts.append(messages)
        return "ما أبعاد مكتب Nexa Pro؟"


def test_follow_up_is_rewritten_answered_in_user_language_and_remembered():
    smart, rewriter = StubSmart(), StubRewriter()
    service = InteractiveSearch(smart, SessionStore(FakeValkey(), 60, 6), rewriter)
    first = service.search("What is the price of the Nexa Pro desk?")
    assert first.new_session and not first.rewrite.applied and first.turn == 1 and first.rewritten_query == first.query
    second = service.search("وكم أبعاده؟", first.session_id)
    assert second.turn == 2 and not second.new_session
    assert second.rewrite.applied and second.rewritten_query == "ما أبعاد مكتب Nexa Pro؟"
    assert second.language_detected == "ar" and second.status == "answered" and second.answer == "answer [1]"
    assert smart.calls[-1] == ("ما أبعاد مكتب Nexa Pro؟", "ar")
    assert "What is the price of the Nexa Pro desk?" in rewriter.prompts[0][1]["content"]
    history = service.store.history(first.session_id)
    assert [item.standalone_query for item in history] == ["What is the price of the Nexa Pro desk?", "ما أبعاد مكتب Nexa Pro؟"]
    assert {"memory_read", "query_rewrite", "retrieval", "rerank", "time_to_first_token", "generation_total", "smart_search", "memory_write", "total"} <= set(second.latency_ms)
    assert second.latency_ms["time_to_first_token"] >= 3.0
    assert second.suggested_followups == []


def test_rewrite_falls_back_to_concatenation_when_generator_fails():
    import httpx

    class FailingRewriter(StubRewriter):
        def complete(self, messages):
            raise httpx.ConnectError("down")

    result = FailingRewriter().rewrite("And its dimensions?", [turn()])
    assert result.applied and result.fallback
    assert result.query == "What is the price of the Nexa Pro desk? And its dimensions?"


def test_v1_prompt_is_unchanged_and_new_examples_avoid_knowledge_bank_names():
    import json
    import re

    from rag.config import REPO_ROOT
    from rag.rewrite import REWRITE_PROMPTS

    assert "وماذا عن طاولة Sol؟" in REWRITE_PROMPTS["v1"] and "كم رقم جوال فرع تبوك؟" in REWRITE_PROMPTS["v1"]
    assert REWRITE_PROMPTS["v2"].endswith(REWRITE_PROMPTS["v3"].split("\n\n", 1)[1])
    corpus = (REPO_ROOT / "data" / "corpus" / "processed" / "chunks.jsonl").read_text(encoding="utf-8").lower()
    for name in ("Orbit", "Luna", "Nova", "Cedar", "Zahra", "Yara", "Dunes", "Falcon", "ينبع"):
        assert name in REWRITE_PROMPTS["v2"]
        assert not re.search(rf"\b{re.escape(name.lower())}\b", corpus), name


def test_suggested_followups_are_returned_in_the_user_language():
    service = InteractiveSearch(StubSmart(), SessionStore(FakeValkey(), 60, 6), StubRewriter(), StubSuggester())
    english = service.search("What is the price of the Nexa Pro desk?")
    assert english.suggested_followups == ["What is the warranty on the Nexa Pro desk?"]
    assert "followups" in english.latency_ms and "followups_wait" in english.latency_ms
    arabic = service.search("وكم أبعاده؟", english.session_id)
    assert arabic.suggested_followups == ["ما مدة ضمان مكتب Nexa Pro؟"]


def test_parse_followups_drops_numbering_repeats_and_wrong_language():
    text = (
        "1. What is the warranty on the Nexa Pro desk?\n"
        "2) ما سعر مكتب Nexa Pro؟\n"
        "- What is the price of the Nexa Pro desk?\n"
        "- How long does delivery to Riyadh take?\n"
        "- How long does delivery to Riyadh take?"
    )
    assert parse_followups(text, "What is the price of the Nexa Pro desk?", "en", 2) == [
        "What is the warranty on the Nexa Pro desk?",
        "How long does delivery to Riyadh take?",
    ]
    assert parse_followups("ما سعر مكتب Nexa Pro؟\nWhat is the price?", "كم أبعاده؟", "ar", 2) == ["ما سعر مكتب Nexa Pro؟"]


def test_followup_suggester_returns_nothing_when_generator_fails():
    import httpx

    class Failing(FollowupSuggester):
        def complete(self, messages):
            raise httpx.ConnectError("down")

    chunk = Chunk(
        chunk_id="desk_en#1", doc_id="desk_en", position=0, title="Desks", section=None, page=None,
        language="en", domain="catalog", format="txt", source="catalog/desk_en.txt", text="The Nexa Pro desk costs 2,980 SAR.",
    )
    assert Failing(CONFIG.generation, CONFIG.interactive).suggest("q?", "en", [chunk]) == []
