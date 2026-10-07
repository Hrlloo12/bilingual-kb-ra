from __future__ import annotations

import argparse
import asyncio
import json
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import numpy as np

from rag.config import REPO_ROOT
from rag.evaluation.metrics import latency_summary


class Sampler:
    def __init__(self, interval_s: float = 1.0) -> None:
        self.interval_s = interval_s
        self.gpu_util: list[float] = []
        self.gpu_memory: list[float] = []
        self.cpu: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        import psutil

        psutil.cpu_percent(None)
        while not self._stop.wait(self.interval_s):
            self.cpu.append(psutil.cpu_percent(None))
            try:
                output = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, check=True, timeout=5,
                ).stdout.strip().splitlines()[0]
                utilization, memory = (float(value) for value in output.split(","))
                self.gpu_util.append(utilization)
                self.gpu_memory.append(memory)
            except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
                continue

    def __enter__(self) -> Sampler:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()

    def summary(self) -> dict:
        def stats(values: list[float]) -> dict:
            return {"avg": round(float(np.mean(values)), 1), "max": round(float(np.max(values)), 1)} if values else {}

        return {"gpu_util_percent": stats(self.gpu_util), "gpu_memory_mib": stats(self.gpu_memory), "cpu_percent": stats(self.cpu)}


async def run_level(base_url: str, mode: str, queries: list[str], users: int, requests: int, timeout_s: float) -> dict:
    latencies, statuses, failures = [], {}, []
    cursor = 0
    lock = asyncio.Lock()

    async def worker(client: httpx.AsyncClient) -> None:
        nonlocal cursor
        while True:
            async with lock:
                if cursor >= requests:
                    return
                query = queries[cursor % len(queries)]
                cursor += 1
            started = time.perf_counter()
            try:
                response = await client.post("/v1/search", json={"mode": mode, "query": query})
                elapsed = (time.perf_counter() - started) * 1000
                if response.status_code == 200:
                    latencies.append(elapsed)
                    status = response.json().get("status", "ok")
                    statuses[status] = statuses.get(status, 0) + 1
                else:
                    failures.append(f"http {response.status_code}")
            except httpx.HTTPError as error:
                failures.append(type(error).__name__)

    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_s) as client:
        with Sampler() as sampler:
            started = time.perf_counter()
            await asyncio.gather(*(worker(client) for _ in range(users)))
            wall = time.perf_counter() - started
    return {
        "users": users,
        "requests": requests,
        "completed": len(latencies),
        "failures": len(failures),
        "failure_kinds": sorted(set(failures)),
        "wall_seconds": round(wall, 2),
        "qps": round(len(latencies) / wall, 3),
        "latency_ms": latency_summary(latencies) if latencies else {},
        "statuses": statuses,
        "resources": sampler.summary(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure latency, throughput and resource use under concurrent users.")
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--mode", default="smart_search", choices=["quick_search", "smart_search"])
    parser.add_argument("--users", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--requests-per-user", type=int, default=16)
    parser.add_argument("--min-requests", type=int, default=48)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = [json.loads(line) for line in (REPO_ROOT / "data" / "test.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    queries = [row["query"] for row in rows]
    random.Random(args.seed).shuffle(queries)
    asyncio.run(run_level(args.base_url, args.mode, queries[-8:], 2, 8, args.timeout))

    unanswerable_share = sum(1 for row in rows if not row["relevant_fact_ids"]) / len(rows)
    report = {
        "mode": args.mode,
        "query_pool": f"data/test.jsonl shuffled with seed {args.seed}; every level starts from the same position, so all levels see the same question mix",
        "pool_unanswerable_share": round(unanswerable_share, 3),
        "levels": [],
    }
    for users in args.users:
        requests = max(args.min_requests, users * args.requests_per_user)
        level = asyncio.run(run_level(args.base_url, args.mode, queries, users, requests, args.timeout))
        report["levels"].append(level)
        print(json.dumps({key: level[key] for key in ("users", "completed", "failures", "qps", "latency_ms")}))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
