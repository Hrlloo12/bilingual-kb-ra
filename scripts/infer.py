from __future__ import annotations

import argparse
import sys

from rag.config import load_serving_config
from rag.quick_search import QuickSearch

MODES = ("quick_search", "smart_ai_search", "smart_search", "interactive")


def interactive_search(config, query: str, session: str):
    from rag.followups import FollowupSuggester
    from rag.interactive import InteractiveSearch
    from rag.rewrite import QueryRewriter
    from rag.sessions import SessionStore
    from rag.smart_search import SmartSearch

    store = SessionStore.from_url(config.valkey.url, config.interactive.session_ttl_s, config.interactive.max_turns)
    service = InteractiveSearch(
        SmartSearch(config),
        store,
        QueryRewriter(config.generation, config.interactive),
        FollowupSuggester(config.generation, config.interactive),
    )
    return service.search(query, None if session == "new" else session)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a query against the knowledge bank in any of the three modes.")
    parser.add_argument("--mode", choices=MODES, required=True, help="smart_search is accepted as an alias of smart_ai_search.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--session", default="new", help="Interactive mode: 'new' starts a conversation; pass the returned session_id to continue it.")
    parser.add_argument("--top-k", type=int, default=None, help="Quick Search: number of results.")
    args = parser.parse_args(argv)

    config = load_serving_config()
    if args.mode == "quick_search":
        response = QuickSearch(config).search(args.query, args.top_k)
    elif args.mode == "interactive":
        response = interactive_search(config, args.query, args.session)
    else:
        from rag.smart_search import SmartSearch

        response = SmartSearch(config).search(args.query)
    sys.stdout.reconfigure(encoding="utf-8")
    print(response.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
