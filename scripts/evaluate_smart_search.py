from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from rag.config import REPO_ROOT, load_serving_config
from rag.evaluation.answers import number_set, summarize_by_bucket
from rag.evaluation.relevance import chunk_fact_labels, load_queries, relevant_chunk_ids
from rag.ingestion import CHUNKS_FILE_NAME, MANIFEST_NAME, read_chunks


def gpu_memory_used_mib() -> list[int]:
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return [int(line) for line in output.split() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Smart AI Search end to end over evaluation queries.")
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--label", default="smart_search")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "smart_search")
    parser.add_argument("--abstain-threshold", type=float, default=None)
    args = parser.parse_args(argv)

    import torch

    from rag.smart_search import SmartSearch

    config = load_serving_config()
    if args.abstain_threshold is not None:
        config = config.model_copy(update={"smart_search": config.smart_search.model_copy(update={"abstain_threshold": args.abstain_threshold})})
    chunks = read_chunks(config.paths.corpus_processed / CHUNKS_FILE_NAME)
    labels = chunk_fact_labels(chunks, config.paths.corpus_raw / MANIFEST_NAME)
    memory_before = gpu_memory_used_mib()
    search = SmartSearch(config)
    for warmup in ("ما هي مدة ضمان المطابخ؟", "What is the return window?", "كم سعر الـ desk؟"):
        search.search(warmup)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "label": args.label,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "config": {
            "embedding": config.embedding.model,
            "reranker": config.reranker.model,
            "generator": config.generation.model,
            "retrieval": config.retrieval.model_dump(),
            "smart_search": config.smart_search.model_dump(),
            "generation": {key: value for key, value in config.generation.model_dump().items() if key != "url"},
        },
        "gpu_memory_used_mib_before_load": memory_before,
        "splits": {},
    }
    for split in args.splits:
        queries = load_queries(REPO_ROOT / "data" / f"{split}.jsonl")[: args.limit]
        references = {
            row["id"]: row.get("reference_answer", "")
            for row in map(json.loads, (REPO_ROOT / "data" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines())
        }
        records = []
        for query in queries:
            response = search.search(query.query)
            relevant = sorted(relevant_chunk_ids(query, labels))
            context_ids = [passage.chunk_id for passage in response.retrieved if passage.in_context]
            reference = references[query.id] if query.answerable else ""
            records.append(
                {
                    "id": query.id,
                    "bucket": query.bucket,
                    "language": query.language,
                    "answerable": query.answerable,
                    "query": query.query,
                    "reference_answer": reference or "NOT_FOUND",
                    "status": response.status,
                    "answer": response.answer,
                    "abstain_reason": response.abstain_reason,
                    "cited_chunk_ids": [citation.chunk_id for citation in response.citations],
                    "context_chunk_ids": context_ids,
                    "relevant_chunk_ids": relevant,
                    "context_hit": any(chunk_id in relevant for chunk_id in context_ids),
                    "top_rerank_score": response.retrieved[0].rerank_score if response.retrieved else None,
                    "post_checks": response.post_checks,
                    "reference_numbers": number_set(reference),
                    "answer_numbers": number_set(response.answer) if response.status == "answered" else [],
                    "usage": response.usage,
                    "latency_ms": response.latency_ms,
                }
            )
        with (args.output_dir / f"{args.label}_{split}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        report["splits"][split] = summarize_by_bucket(records)
        print(json.dumps({split: report["splits"][split]["overall"]}, indent=2, ensure_ascii=False))

    report["gpu_memory_used_mib_after_run"] = gpu_memory_used_mib()
    if torch.cuda.is_available():
        report["torch_peak_allocated_mib_embedder_reranker"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    (args.output_dir / f"{args.label}_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
