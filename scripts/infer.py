from __future__ import annotations

import argparse
import sys

from rag.config import load_serving_config
from rag.quick_search import QuickSearch

MODES = ("quick_search",)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a query against the knowledge bank.")
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args(argv)

    config = load_serving_config()
    if args.mode == "quick_search":
        response = QuickSearch(config).search(args.query, args.top_k)
    sys.stdout.reconfigure(encoding="utf-8")
    print(response.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
