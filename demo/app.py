from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent
QUERIES = DEMO_DIR / "demo_queries.json"
SUMMARY = REPO_ROOT / "results" / "summary.json"
STAGES = (
    ("query_rewrite", "rewrite"),
    ("preprocessing", "preprocessing"),
    ("search", "search"),
    ("retrieval", "retrieval"),
    ("rerank", "rerank"),
    ("time_to_first_token", "first token"),
    ("generation_total", "generation"),
    ("total", "total"),
)


def latency_line(latency: dict) -> str:
    return " · ".join(f"{label} {latency[key]:.0f} ms" for key, label in STAGES if key in latency)


def show(step: dict, body: dict, elapsed_ms: float) -> None:
    print(f"\n=== {step['label']}")
    print(f"[{body['mode']}] {body['query']}")
    if body["mode"] == "quick_search":
        for rank, result in enumerate(body["results"][:3], start=1):
            print(f"  {rank}. {result['title']} › {result.get('section') or ''} (score {result['score']:.2f}, {result['source']}, page {result['page']})")
            print(f"     {result['snippet'][:160]}")
    else:
        if body["mode"] == "interactive":
            print(f"  session {body['session_id'][:8]} · turn {body['turn']} · searched as: {body['rewritten_query']}")
        print(f"  answer ({body['language_detected']}): {body['answer']}")
        for citation in body["citations"]:
            print(f"  [{citation['id']}] {citation['doc_id']} · {citation['chunk_id']} · page {citation['page']} · relevance {citation['relevance']}")
        if body.get("abstain_reason"):
            print(f"  NOT_FOUND reason: {body['abstain_reason']}")
        for question in body.get("suggested_followups", []):
            print(f"  suggested: {question}")
    print(f"  latency: {latency_line(body['latency_ms'])} · client {elapsed_ms:.0f} ms")


def run(args: argparse.Namespace) -> int:
    plan = json.loads(QUERIES.read_text(encoding="utf-8"))
    steps = plan["steps"] if args.set == "steps" else plan["after_ingestion"] if args.set == "after_ingestion" else plan["steps"] + plan["after_ingestion"]
    client = httpx.Client(base_url=args.base_url, timeout=args.timeout)
    for mode in ("quick_search", "smart_ai_search"):
        client.post("/v1/search", json={"mode": mode, "query": "What is the return window?"})
    sessions: dict[str, str] = {}
    transcript = []
    for step in steps:
        body = {"mode": step["mode"], "query": step["query"]}
        if step.get("conversation") in sessions:
            body["session_id"] = sessions[step["conversation"]]
        started = time.perf_counter()
        response = client.post("/v1/search", json=body)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        data = response.json()
        if step.get("conversation"):
            sessions[step["conversation"]] = data["session_id"]
        show(step, data, elapsed_ms)
        transcript.append({"step": step, "client_ms": round(elapsed_ms, 1), "response": data})
        if args.pause:
            input("  (press Enter for the next query)")
    for session_id in sessions.values():
        client.delete(f"/v1/sessions/{session_id}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(transcript, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


def table(title: str, rows: list[dict], columns: list[str]) -> None:
    print(f"\n{title}")
    widths = [max(len(column), *(len(str(row.get(column, ""))) for row in rows)) for column in columns]
    print("  ".join(column.ljust(width) for column, width in zip(columns, widths)))
    for row in rows:
        print("  ".join(str(row.get(column, "")).ljust(width) for column, width in zip(columns, widths)))


def results(args: argparse.Namespace) -> int:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    table("Before vs after fine-tuning (held-out test split, NVIDIA L4)", summary["before_vs_after"], ["metric", "before", "after"])
    table("Latency, one user at a time (ms)", summary["latency"], ["mode", "avg", "p50", "p95", "max", "first_token_p95"])
    table("Concurrency (p95 ms / requests per second)", summary["concurrency"], ["users", "quick_search", "smart_ai_search", "interactive"])
    return 0


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Live demo of the three search modes against a running API, and a summary of measured results.")
    commands = parser.add_subparsers(dest="command", required=True)
    play = commands.add_parser("run", help="Send the demo queries in demo/demo_queries.json to the API and print answers, citations and latency.")
    play.add_argument("--base-url", default="http://localhost:8080")
    play.add_argument("--set", choices=("steps", "after_ingestion", "all"), default="steps")
    play.add_argument("--pause", action="store_true", help="Wait for Enter between queries, for recording.")
    play.add_argument("--timeout", type=float, default=120.0)
    play.add_argument("--output", type=Path, default=None)
    commands.add_parser("results", help="Print the before/after, latency and concurrency tables from results/summary.json.")
    args = parser.parse_args(argv)
    return run(args) if args.command == "run" else results(args)


if __name__ == "__main__":
    sys.exit(main())
