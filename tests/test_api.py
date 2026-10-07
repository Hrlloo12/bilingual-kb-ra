import httpx
import pytest
from fastapi.testclient import TestClient

from rag.api import Components, create_app
from rag.interactive import InteractiveSearch
from rag.schemas import QuickSearchResponse, SmartSearchResponse
from rag.sessions import SessionStore
from test_interactive import FakeValkey, StubRewriter, StubSmart


class StubQuick:
    def search(self, query, top_k=None):
        return QuickSearchResponse(query=query, language_detected="en", results=[], latency_ms={"total": 1.0})


class DownSmart(StubSmart):
    def search(self, query, language=None):
        raise httpx.ConnectError("generator down")


def make_client(smart=None, checks=None):
    smart = smart or StubSmart()
    store = SessionStore(FakeValkey(), 60, 6)
    components = Components(
        quick=StubQuick(),
        smart=smart,
        interactive=InteractiveSearch(smart, store, StubRewriter()),
        store=store,
        checks=checks or {"opensearch": lambda: True, "generator": lambda: True},
    )
    return TestClient(create_app(components))


def test_all_three_modes_share_one_endpoint():
    with make_client() as client:
        assert client.post("/v1/search", json={"mode": "quick_search", "query": "returns"}).json()["mode"] == "quick_search"
        smart = client.post("/v1/search", json={"mode": "smart_search", "query": "What is the return window?"}).json()
        assert smart["mode"] == "smart_search" and smart["status"] == "answered"
        first = client.post("/v1/search", json={"mode": "interactive", "query": "What is the price of the Nexa Pro desk?"}).json()
        second = client.post("/v1/search", json={"mode": "interactive", "query": "وكم أبعاده؟", "session_id": first["session_id"]}).json()
        assert second["turn"] == 2 and second["rewrite"]["applied"]
        session = client.get(f"/v1/sessions/{first['session_id']}").json()
        assert len(session["turns"]) == 2 and session["ttl_s"] == 60
        assert client.delete(f"/v1/sessions/{first['session_id']}").json()["deleted"]
        assert client.get(f"/v1/sessions/{first['session_id']}").status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {"mode": "smart_search", "query": "   "},
        {"mode": "unknown", "query": "x"},
        {"mode": "interactive", "query": "x", "session_id": "bad id!"},
        {"mode": "smart_search", "query": "x" * 1001},
    ],
)
def test_invalid_requests_are_rejected(body):
    with make_client() as client:
        assert client.post("/v1/search", json=body).status_code == 422


def test_generator_outage_returns_503():
    with make_client(smart=DownSmart()) as client:
        response = client.post("/v1/search", json={"mode": "smart_search", "query": "What is the return window?"})
        assert response.status_code == 503
        assert response.json()["detail"] == "generator unavailable"


def test_health_reports_each_component():
    with make_client(checks={"opensearch": lambda: True, "generator": lambda: False}) as client:
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json()["components"] == {"opensearch": "ok", "generator": "not_ready"}
        assert client.get("/livez").json() == {"status": "ok"}
        assert "Qimam Knowledge Search" in client.get("/").text
