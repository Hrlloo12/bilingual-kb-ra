from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from rag.config import load_serving_config
from rag.facts import FactBase
from rag.kb_render import check_coverage, load_templates, render_document, write_corpus


def validate_facts(facts_dir: Path) -> int:
    fact_base = FactBase.load(facts_dir)
    summary = fact_base.summary()
    totals = {key: sum(domain[key] for domain in summary.values()) for key in next(iter(summary.values()))}
    print(json.dumps({"domains": summary, "totals": totals}, indent=2))
    return 0


def render_kb(facts_dir: Path, templates_dir: Path, corpus_dir: Path) -> int:
    fact_base = FactBase.load(facts_dir)
    documents = [render_document(template, fact_base) for template in load_templates(templates_dir)]
    errors = check_coverage(documents, fact_base)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    write_corpus(documents, corpus_dir)
    report = {
        "documents": len(documents),
        "by_language": dict(Counter(document.language for document in documents)),
        "by_format": dict(Counter(document.format for document in documents)),
        "eastern_digit_documents": sum(1 for document in documents if document.digits == "eastern"),
        "sections": sum(len(document.sections) for document in documents),
        "facts_rendered": len({fact_id for document in documents for section in document.sections for fact_id in section.fact_ids}),
        "facts_total": len(fact_base.facts),
        "corpus_dir": str(corpus_dir),
    }
    print(json.dumps(report, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the knowledge bank corpus and datasets.")
    parser.add_argument("--facts-dir", type=Path, default=None)
    parser.add_argument("--templates-dir", type=Path, default=None)
    parser.add_argument("--corpus-dir", type=Path, default=None)
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("validate-facts", help="Validate fact files and print coverage statistics.")
    subcommands.add_parser("render-kb", help="Render document templates into the raw knowledge bank corpus.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = load_serving_config().paths
    facts_dir = args.facts_dir or paths.facts_dir
    if args.command == "validate-facts":
        return validate_facts(facts_dir)
    if args.command == "render-kb":
        return render_kb(facts_dir, args.templates_dir or paths.templates_dir, args.corpus_dir or paths.corpus_raw)
    return 1


if __name__ == "__main__":
    sys.exit(main())
