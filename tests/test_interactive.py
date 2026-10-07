import pytest

from rag.config import load_serving_config
from rag.interactive import InteractiveSearch
from rag.rewrite import QueryRewriter, build_rewrite_messages, clean_rewrite, gate
from rag.schemas import SmartSearchResponse
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

    def search(self, query, language=None):
        self.calls.append((query, language))
        return SmartSearchResponse(
            query=query,
            language_detected=language or "en",
            status="answered",
            answer="answer [1]",
            citations=[],
            retrieved=[],
            latency_ms={"total": 5.0},
        )


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
    assert first.new_session and not first.rewrite.applied and first.turn == 1
    second = service.search("وكم أبعاده؟", first.session_id)
    assert second.turn == 2 and not second.new_session
    assert second.rewrite.applied and second.rewrite.standalone_query == "ما أبعاد مكتب Nexa Pro؟"
    assert smart.calls[-1] == ("ما أبعاد مكتب Nexa Pro؟", "ar")
    assert "What is the price of the Nexa Pro desk?" in rewriter.prompts[0][1]["content"]
    history = service.store.history(first.session_id)
    assert [item.standalone_query for item in history] == ["What is the price of the Nexa Pro desk?", "ما أبعاد مكتب Nexa Pro؟"]
    assert set(second.latency_ms) == {"memory_read", "rewrite", "smart_search", "memory_write", "total"}


def test_rewrite_falls_back_to_concatenation_when_generator_fails():
    import httpx

    class FailingRewriter(StubRewriter):
        def complete(self, messages):
            raise httpx.ConnectError("down")

    result = FailingRewriter().rewrite("And its dimensions?", [turn()])
    assert result.applied and result.fallback
    assert result.query == "What is the price of the Nexa Pro desk? And its dimensions?"
