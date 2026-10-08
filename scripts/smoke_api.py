from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

SMART_ANSWERABLE = [
    "ما هي سياسة الاسترجاع للطلبات المتأخرة؟",
    "What is the SLA for premium support tickets?",
    "أبغى أعرف الـ pricing حق الباقة المؤسسية",
]
UNANSWERABLE = "Do you offer furniture rental for events and exhibitions?"
CONVERSATIONS = [
    ("What is the price of the Nexa Pro standing desk?", "And what are its dimensions?"),
    ("ما سعر سرير Aria؟", "وماذا عن سرير Noor؟"),
    ("Where is the Abha showroom?", "وش رقم الـ phone حقه؟"),
]


class Smoke:
    def __init__(self, base_url: str, timeout_s: float) -> None:
        self.client = httpx.Client(base_url=base_url, timeout=timeout_s)
        self.checks: list[dict] = []

    def check(self, name: str, passed: bool, **details) -> None:
        self.checks.append({"name": name, "passed": bool(passed), **details})
        print(f"{'PASS' if passed else 'FAIL'} {name} {json.dumps(details, ensure_ascii=False)[:300]}")

    def search(self, body: dict) -> tuple[int, dict, float]:
        started = time.perf_counter()
        response = self.client.post("/v1/search", json=body)
        return response.status_code, response.json(), round((time.perf_counter() - started) * 1000, 1)

    def run(self) -> bool:
        health = self.client.get("/health")
        self.check("health", health.status_code == 200, body=health.json())
        self.check("ui", self.client.get("/").status_code == 200 and "Qimam Knowledge Search" in self.client.get("/").text)

        status, body, elapsed = self.search({"mode": "quick_search", "query": "سياسة الاسترجاع"})
        self.check("quick_search", status == 200 and len(body.get("results", [])) > 0, results=len(body.get("results", [])), ms=elapsed)

        for query in SMART_ANSWERABLE:
            status, body, elapsed = self.search({"mode": "smart_ai_search", "query": query})
            self.check(
                f"smart_ai_search answered: {query}",
                status == 200 and body.get("status") == "answered" and len(body.get("citations", [])) > 0,
                answer=body.get("answer"),
                sources=[citation["source"] for citation in body.get("citations", [])],
                top_rerank_score=(body.get("retrieved") or [{}])[0].get("rerank_score"),
                abstain_reason=body.get("abstain_reason"),
                ms=elapsed,
            )

        status, body, elapsed = self.search({"mode": "smart_ai_search", "query": UNANSWERABLE})
        self.check("smart_ai_search NOT_FOUND", status == 200 and body.get("status") == "not_found" and body.get("answer") == "NOT_FOUND", reason=body.get("abstain_reason"), ms=elapsed)

        for first, follow_up in CONVERSATIONS:
            status, opening, _ = self.search({"mode": "interactive", "query": first})
            session_id = opening.get("session_id")
            status_two, second, elapsed = self.search({"mode": "interactive", "query": follow_up, "session_id": session_id})
            rewrite = second.get("rewrite", {})
            self.check(
                f"interactive follow-up: {follow_up}",
                status == 200 and status_two == 200 and second.get("turn") == 2 and rewrite.get("applied"),
                standalone=second.get("rewritten_query"),
                answer=second.get("answer"),
                suggested_followups=second.get("suggested_followups"),
                ms=elapsed,
            )
            self.check("interactive suggested follow-ups", len(second.get("suggested_followups", [])) > 0, suggested_followups=second.get("suggested_followups"))
            history = self.client.get(f"/v1/sessions/{session_id}")
            self.check("session stored", history.status_code == 200 and len(history.json()["turns"]) == 2, ttl_s=history.json().get("ttl_s"))
            self.check("session deleted", self.client.delete(f"/v1/sessions/{session_id}").json()["deleted"])

        status, _, _ = self.search({"mode": "smart_ai_search", "query": "  "})
        self.check("blank query rejected", status == 422)
        return all(item["passed"] for item in self.checks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test a running deployment through its public API.")
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)
    smoke = Smoke(args.base_url, args.timeout)
    passed = smoke.run()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"passed": passed, "checks": smoke.checks}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{sum(item['passed'] for item in smoke.checks)}/{len(smoke.checks)} checks passed")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
