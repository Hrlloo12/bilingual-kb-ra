from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag.config import load_serving_config
from rag.facts import FactBase


def validate_facts(facts_dir: Path) -> int:
    fact_base = FactBase.load(facts_dir)
    summary = fact_base.summary()
    totals = {key: sum(domain[key] for domain in summary.values()) for key in next(iter(summary.values()))}
    print(json.dumps({"domains": summary, "totals": totals}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the knowledge bank corpus and datasets.")
    parser.add_argument("--facts-dir", type=Path, default=None)
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("validate-facts", help="Validate fact files and print coverage statistics.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = load_serving_config().paths
    facts_dir = args.facts_dir or paths.facts_dir
    if args.command == "validate-facts":
        return validate_facts(facts_dir)
    return 1


if __name__ == "__main__":
    sys.exit(main())
