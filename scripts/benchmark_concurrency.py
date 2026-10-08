from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import threading
import time
from pathlib import Path

import httpx
import numpy as np

from rag.config import REPO_ROOT
from rag.evaluation.hardware import gpu_snapshot, host_info
from rag.evaluation.metrics import latency_summary

MODES = ("quick_search", "smart_ai_search", "interactive")


class Sampler:
    def __init__(self, interval_s: float = 1.0) -> None:
        self.interval_s = interval_s
        self.gpu_util: list[float] = []
        self.gpu_memory: list[float] = []
        self.cpu: list[float] = []
        self.ram: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        import psutil

        self.cpu.append(psutil.cpu_percent(None))
        memory = psutil.virtual_memory()
        self.ram.append((memory.total - memory.available) / 2**20)
        snapshot = gpu_snapshot()
        if snapshot:
            self.gpu_util.append(snapshot["utilization_percent"])
            self.gpu_memory.append(snapshot["memory_used_mib"])

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def __enter__(self) -> Sampler:
        import psutil

        psutil.cpu_percent(None)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()

    def summary(self) -> dict:
        def stats(values: list[float]) -> dict:
            return {"avg": round(float(np.mean(values)), 1), "max": round(float(np.max(values)), 1)} if values else {}

        return {
            "gpu_util_percent": stats(self.gpu_util),
            "gpu_memory_mib": stats(self.gpu_memory),
            "cpu_percent": stats(self.cpu),
            "host_ram_used_mib": stats(self.ram),
        }


def load_work(mode: str, seed: int) -> tuple[list, float]:
    if mode == "interactive":
        path = REPO_ROOT / "data" / "interactive" / "conversations_test.jsonl"
        conversations = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        random.Random(seed).shuffle(conversations)
        return [[turn["query"] for turn in conversation["turns"]] for conversation in conversations], 0.0
    rows = [json.loads(line) for line in (REPO_ROOT / "data" / "test.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    queries = [[row["query"]] for row in rows]
    random.Random(seed).shuffle(queries)
    return queries, sum(1 for row in rows if not row["relevant_fact_ids"]) / len(rows)


async def run_level(base_url: str, mode: str, work: list[list[str]], users: int, requests: int, timeout_s: float) -> dict:
    latencies, follow_up_latencies, statuses, failures = [], [], {}, []
    server_stages: dict[str, list[float]] = {}
    sent = 0
    cursor = 0
    lock = asyncio.Lock()

    async def send(client: httpx.AsyncClient, body: dict) -> dict | None:
        started = time.perf_counter()
        try:
            response = await client.post("/v1/search", json=body)
        except httpx.HTTPError as error:
            failures.append(type(error).__name__)
            return None
        elapsed = (time.perf_counter() - started) * 1000
        if response.status_code != 200:
            failures.append(f"http {response.status_code}")
            return None
        data = response.json()
        latencies.append(elapsed)
        if data.get("turn", 1) > 1:
            follow_up_latencies.append(elapsed)
        status = data.get("status", "ok")
        statuses[status] = statuses.get(status, 0) + 1
        for stage, value in data.get("latency_ms", {}).items():
            server_stages.setdefault(stage, []).append(value)
        return data

    async def worker(client: httpx.AsyncClient) -> None:
        nonlocal cursor, sent
        while True:
            async with lock:
                if sent >= requests:
                    return
                turns = work[cursor % len(work)]
                cursor += 1
                sent += len(turns)
            session_id = None
            for query in turns:
                body = {"mode": mode, "query": query}
                if session_id:
                    body["session_id"] = session_id
                data = await send(client, body)
                if data is None:
                    break
                session_id = data.get("session_id")
            if session_id:
                await client.delete(f"/v1/sessions/{session_id}")

    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_s) as client:
        with Sampler() as sampler:
            started = time.perf_counter()
            await asyncio.gather(*(worker(client) for _ in range(users)))
            wall = time.perf_counter() - started
    level = {
        "users": users,
        "requests": sent,
        "completed": len(latencies),
        "failures": len(failures),
        "failure_rate": round(len(failures) / sent, 4) if sent else 0.0,
        "failure_kinds": sorted(set(failures)),
        "wall_seconds": round(wall, 2),
        "qps": round(len(latencies) / wall, 3),
        "latency_ms": latency_summary(latencies) if latencies else {},
        "server_latency_ms": {stage: latency_summary(values) for stage, values in sorted(server_stages.items())},
        "statuses": statuses,
        "resources": sampler.summary(),
    }
    if follow_up_latencies:
        level["follow_up_latency_ms"] = latency_summary(follow_up_latencies)
    return level


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure latency, throughput and resource use under concurrent users.")
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--mode", default="smart_ai_search", choices=MODES)
    parser.add_argument("--users", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--requests-per-user", type=int, default=16)
    parser.add_argument("--min-requests", type=int, default=48)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    work, unanswerable_share = load_work(args.mode, args.seed)
    asyncio.run(run_level(args.base_url, args.mode, work[-4:], 2, 8, args.timeout))

    report = {
        "mode": args.mode,
        "query_pool": (
            f"{'data/interactive/conversations_test.jsonl (two turns per conversation, one session each)' if args.mode == 'interactive' else 'data/test.jsonl'}"
            f" shuffled with seed {args.seed}; every level starts from the same position, so all levels see the same mix"
        ),
        "pool_unanswerable_share": round(unanswerable_share, 3),
        "host": host_info(),
        "levels": [],
    }
    for users in args.users:
        requests = max(args.min_requests, users * args.requests_per_user)
        level = asyncio.run(run_level(args.base_url, args.mode, work, users, requests, args.timeout))
        report["levels"].append(level)
        print(json.dumps({key: level[key] for key in ("users", "completed", "failures", "qps", "latency_ms")}))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
