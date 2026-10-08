from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import httpx

from rag.config import REPO_ROOT, load_serving_config
from rag.evaluation.hardware import host_info
from rag.evaluation.metrics import latency_summary

MODES = ("quick_search", "smart_ai_search", "interactive")
FIRST_RESULT_STAGES = {"quick_search": ("preprocessing", "search"), "smart_ai_search": ("preprocessing", "retrieval")}
MAX_QUERY_CHARS = 1000


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_queries(seed: int) -> list[dict]:
    rows = read_jsonl(REPO_ROOT / "data" / "test.jsonl")
    random.Random(seed).shuffle(rows)
    return rows


def long_queries(rows: list[dict]) -> dict[str, str]:
    by_length = sorted(rows, key=lambda row: -len(row["query"]))
    longest_ar = next(row["query"] for row in by_length if row["language"] != "en")
    longest_en = next(row["query"] for row in by_length if row["language"] == "en")
    parts, total = [], 0
    for row in by_length:
        if row["language"] == "en" and total + len(row["query"]) + 1 <= MAX_QUERY_CHARS:
            parts.append(row["query"])
            total += len(row["query"]) + 1
    return {"longest_arabic_test_query": longest_ar, "longest_english_test_query": longest_en, "max_length_input": " ".join(parts)}


class Recorder:
    def __init__(self) -> None:
        self.client_ms: list[float] = []
        self.stages: dict[str, list[float]] = {}
        self.first_result: list[float] = []
        self.first_token: list[float] = []
        self.prompt_tokens: list[int] = []

    def add(self, mode: str, elapsed_ms: float, body: dict) -> None:
        self.client_ms.append(elapsed_ms)
        latency = body.get("latency_ms", {})
        for stage, value in latency.items():
            self.stages.setdefault(stage, []).append(value)
        stages = FIRST_RESULT_STAGES.get("smart_ai_search" if mode == "interactive" else mode, ())
        if stages and all(stage in latency for stage in stages):
            offset = latency.get("memory_read", 0.0) + latency.get("query_rewrite", 0.0) if mode == "interactive" else 0.0
            self.first_result.append(offset + sum(latency[stage] for stage in stages))
        if "time_to_first_token" in latency:
            self.first_token.append(latency["time_to_first_token"])
        if body.get("usage", {}).get("prompt_tokens"):
            self.prompt_tokens.append(body["usage"]["prompt_tokens"])

    def summary(self) -> dict:
        report = {
            "requests": len(self.client_ms),
            "client_total_ms": latency_summary(self.client_ms),
            "server_stages_ms": {stage: latency_summary(values) for stage, values in sorted(self.stages.items())},
        }
        if self.first_result:
            report["time_to_first_result_ms"] = latency_summary(self.first_result)
        if self.first_token:
            report["time_to_first_token_ms"] = latency_summary(self.first_token)
        if self.prompt_tokens:
            report["prompt_tokens"] = {"avg": round(sum(self.prompt_tokens) / len(self.prompt_tokens), 1), "max": max(self.prompt_tokens)}
        return report


def run_api(args: argparse.Namespace) -> dict:
    client = httpx.Client(base_url=args.base_url, timeout=args.timeout)

    def post(body: dict) -> tuple[float, dict]:
        started = time.perf_counter()
        response = client.post("/v1/search", json=body)
        elapsed = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        return elapsed, response.json()

    rows = test_queries(args.seed)
    report: dict = {"base_url": args.base_url, "host": host_info(), "query_pool": f"data/test.jsonl shuffled with seed {args.seed}", "modes": {}}
    for mode in args.modes:
        post({"mode": mode, "query": rows[0]["query"]})
        recorder = Recorder()
        if mode == "interactive":
            conversations = read_jsonl(REPO_ROOT / "data" / "interactive" / "conversations_test.jsonl")
            random.Random(args.seed).shuffle(conversations)
            follow_ups = Recorder()
            for conversation in conversations[: args.conversations]:
                session_id = None
                for turn in conversation["turns"]:
                    body = {"mode": mode, "query": turn["query"], **({"session_id": session_id} if session_id else {})}
                    elapsed, data = post(body)
                    session_id = data["session_id"]
                    recorder.add(mode, elapsed, data)
                    if data["turn"] > 1:
                        follow_ups.add(mode, elapsed, data)
                client.delete(f"/v1/sessions/{session_id}")
            report["modes"][mode] = {**recorder.summary(), "follow_up_turns": follow_ups.summary()}
        else:
            limit = len(rows) if mode == "quick_search" else args.queries
            for row in rows[:limit]:
                elapsed, data = post({"mode": mode, "query": row["query"]})
                recorder.add(mode, elapsed, data)
            report["modes"][mode] = recorder.summary()
        print(json.dumps({mode: report["modes"][mode]["client_total_ms"]}))

    inputs = long_queries(rows)
    report["max_input"] = {"api_limit_chars": MAX_QUERY_CHARS, "inputs": {}}
    for name, query in inputs.items():
        entry = {"chars": len(query), "query": query, "modes": {}}
        for mode in args.modes:
            recorder = Recorder()
            for _ in range(args.repeats):
                elapsed, data = post({"mode": mode, "query": query})
                recorder.add(mode, elapsed, data)
                if data.get("session_id"):
                    client.delete(f"/v1/sessions/{data['session_id']}")
            entry["modes"][mode] = recorder.summary()
        report["max_input"]["inputs"][name] = entry
    return report


def run_topk(args: argparse.Namespace) -> dict:
    from rag.ingestion import CHUNKS_FILE_NAME, MANIFEST_NAME, read_chunks
    from rag.evaluation.relevance import chunk_fact_labels, load_queries, relevant_chunk_ids
    from rag.quick_search import QuickSearch
    from rag.reranker import Reranker
    from rag.retrieval.dense import Embedder
    from rag.smart_search import SmartSearch

    config = load_serving_config()
    labels = chunk_fact_labels(read_chunks(config.paths.corpus_processed / CHUNKS_FILE_NAME), config.paths.corpus_raw / MANIFEST_NAME)
    queries = load_queries(REPO_ROOT / "data" / "test.jsonl")
    random.Random(args.seed).shuffle(queries)
    queries = queries[: args.queries]
    embedder, reranker = Embedder(config.embedding), Reranker(config.reranker)

    def measure(retrieved: int, reranked: int) -> dict:
        variant = config.model_copy(
            update={
                "retrieval": config.retrieval.model_copy(update={"candidates": retrieved}),
                "smart_search": config.smart_search.model_copy(update={"context_top_k": reranked}),
            }
        )
        search = SmartSearch(variant, embedder=embedder, reranker=reranker)
        search.search(queries[0].query)
        stages: dict[str, list[float]] = {}
        answered, context_hits, answerable, prompt_tokens = 0, 0, 0, []
        for query in queries:
            response = search.search(query.query)
            for stage, value in response.latency_ms.items():
                stages.setdefault(stage, []).append(value)
            answered += response.status == "answered"
            if query.answerable:
                answerable += 1
                relevant = relevant_chunk_ids(query, labels)
                context_hits += any(passage.in_context and passage.chunk_id in relevant for passage in response.retrieved)
            if response.usage.get("prompt_tokens"):
                prompt_tokens.append(response.usage["prompt_tokens"])
        return {
            "retrieved_top_k": retrieved,
            "reranked_top_k": reranked,
            "queries": len(queries),
            "answered_rate": round(answered / len(queries), 4),
            "context_hit_rate_answerable": round(context_hits / answerable, 4) if answerable else None,
            "prompt_tokens_avg": round(sum(prompt_tokens) / len(prompt_tokens), 1) if prompt_tokens else None,
            "latency_ms": {stage: latency_summary(values) for stage, values in sorted(stages.items())},
        }

    report: dict = {
        "host": host_info(),
        "query_pool": f"first {len(queries)} questions of data/test.jsonl shuffled with seed {args.seed}",
        "notes": {
            "retrieved_top_k": "number of fused candidates sent to the reranker (retrieval.candidates)",
            "reranked_top_k": "maximum number of reranked passages sent to the generator (smart_search.context_top_k)",
            "fixed": "abstention threshold and context_min_score are unchanged from configs/serving_config.yaml",
        },
        "retrieved_sweep": [],
        "reranked_sweep": [],
        "quick_search_sweep": [],
    }
    for retrieved in args.retrieved:
        report["retrieved_sweep"].append(measure(retrieved, config.smart_search.context_top_k))
        print(json.dumps({"retrieved_top_k": retrieved, "rerank": report["retrieved_sweep"][-1]["latency_ms"]["rerank"]}))
    for reranked in args.reranked:
        report["reranked_sweep"].append(measure(config.retrieval.candidates, reranked))
        print(json.dumps({"reranked_top_k": reranked, "total": report["reranked_sweep"][-1]["latency_ms"]["total"]}))
    quick = QuickSearch(config)
    for top_k in args.quick_top_k:
        quick.search(queries[0].query, top_k)
        totals = [quick.search(query.query, top_k).latency_ms["total"] for query in queries]
        report["quick_search_sweep"].append({"top_k": top_k, "latency_ms": {"total": latency_summary(totals)}})
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure per-stage latency for every mode, and latency against top-k.")
    commands = parser.add_subparsers(dest="command", required=True)
    api = commands.add_parser("api", help="Sequential requests through the running API for each mode, plus maximum-length inputs.")
    api.add_argument("--base-url", default="http://localhost:8080")
    api.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    api.add_argument("--queries", type=int, default=100, help="Smart AI Search questions; Quick Search uses the whole test split.")
    api.add_argument("--conversations", type=int, default=50)
    api.add_argument("--repeats", type=int, default=5)
    api.add_argument("--timeout", type=float, default=120.0)
    topk = commands.add_parser("topk", help="In-process Smart AI Search latency against retrieved and reranked top-k (needs the GPU stack).")
    topk.add_argument("--queries", type=int, default=100)
    topk.add_argument("--retrieved", nargs="+", type=int, default=[10, 20, 30, 50])
    topk.add_argument("--reranked", nargs="+", type=int, default=[1, 2, 4, 8])
    topk.add_argument("--quick-top-k", nargs="+", type=int, default=[5, 10, 20, 50])
    for command in (api, topk):
        command.add_argument("--seed", type=int, default=2026)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    report = run_api(args) if args.command == "api" else run_topk(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
