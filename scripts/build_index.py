from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

from rag.config import load_serving_config
from rag.ingestion import KnowledgeBankIndexer, discover, parse_and_chunk


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild the keyword and dense indexes from the knowledge bank corpus.")
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument("--lexical-only", action="store_true", help="Build only the OpenSearch BM25 index.")
    args = parser.parse_args(argv)

    config = load_serving_config()
    corpus_dir = args.corpus_dir or config.paths.corpus_raw
    started = perf_counter()
    chunks, report = parse_and_chunk(discover(corpus_dir), corpus_dir, config.chunking.max_chars)
    parse_seconds = perf_counter() - started
    if not chunks:
        print(json.dumps({"error": f"no supported documents found in {corpus_dir}", **report.as_dict()}, indent=2))
        return 1

    index_started = perf_counter()
    counts = KnowledgeBankIndexer(config, with_dense=not args.lexical_only).rebuild(chunks)
    print(
        json.dumps(
            {
                **report.as_dict(),
                "indexed": counts,
                "embedding_model": None if args.lexical_only else config.embedding.model,
                "parse_and_chunk_seconds": round(parse_seconds, 2),
                "index_seconds": round(perf_counter() - index_started, 2),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())
