from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

from rag.config import load_serving_config
from rag.ingestion import KnowledgeBankIndexer, discover, parse_and_chunk


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Add or replace PDF, DOCX, TXT or HTML documents in the knowledge bank.")
    parser.add_argument("--path", type=Path, required=True, help="A document file or a directory of documents.")
    parser.add_argument("--lexical-only", action="store_true", help="Update only the OpenSearch BM25 index.")
    args = parser.parse_args(argv)

    path = args.path.resolve()
    if not path.exists():
        print(json.dumps({"error": f"{path} does not exist"}))
        return 1
    root = path if path.is_dir() else path.parent
    config = load_serving_config()
    started = perf_counter()
    chunks, report = parse_and_chunk(discover(path), root, config.chunking.max_chars)
    counts = KnowledgeBankIndexer(config, with_dense=not args.lexical_only).upsert(chunks) if chunks else {}
    print(
        json.dumps(
            {**report.as_dict(), "indexed": counts, "seconds": round(perf_counter() - started, 2)},
            indent=2,
            ensure_ascii=False,
        )
    )
    return 1 if report.failures or not chunks else 0


if __name__ == "__main__":
    sys.exit(main())
