from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from rag.config import REPO_ROOT, load_serving_config
from rag.dataset_builder import (
    SPLITS,
    assign_ids,
    assign_splits,
    clean_generated,
    fact_components,
    generation_records,
    read_jsonl,
    write_jsonl,
)
from rag.facts import FactBase
from rag.ingestion import MANIFEST_NAME
from rag.kb_render import check_coverage, load_templates, render_document, write_corpus

DATA_DIR = REPO_ROOT / "data"
SPLITS_FILE = DATA_DIR / "splits" / "fact_splits.json"
GENERATION_INPUT = DATA_DIR / "generation" / "fact_prompts.jsonl"
GENERATION_OUTPUT = DATA_DIR / "generation" / "raw_queries.jsonl"
UNANSWERABLE_FILE = DATA_DIR / "unanswerable_queries.jsonl"
DATASET_REPORT = DATA_DIR / "dataset_report.json"
SPLIT_SEED = 2026


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


def split_facts(facts_dir: Path, corpus_dir: Path) -> int:
    fact_base = FactBase.load(facts_dir)
    components = fact_components(read_jsonl(corpus_dir / MANIFEST_NAME))
    splits = assign_splits(components, fact_base, SPLIT_SEED)
    missing = sorted(set(fact_base.facts) - set(splits))
    if missing:
        print(f"facts missing from manifest: {missing[:10]}", file=sys.stderr)
        return 1
    SPLITS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPLITS_FILE.write_text(json.dumps({"seed": SPLIT_SEED, "splits": splits}, indent=2, sort_keys=True), encoding="utf-8")
    write_jsonl(generation_records(fact_base, splits), GENERATION_INPUT)
    by_split_domain: dict[str, Counter] = {split: Counter() for split in SPLITS}
    for fact_id, split in splits.items():
        by_split_domain[split][fact_base.domain_of[fact_id]] += 1
    print(
        json.dumps(
            {
                "components": len(components),
                "largest_component": max(len(component) for component in components),
                "facts_per_split": {split: sum(counter.values()) for split, counter in by_split_domain.items()},
                "by_split_and_domain": {split: dict(counter) for split, counter in by_split_domain.items()},
                "generation_input": str(GENERATION_INPUT.relative_to(REPO_ROOT)),
            },
            indent=2,
        )
    )
    return 0


def build_datasets(facts_dir: Path) -> int:
    fact_base = FactBase.load(facts_dir)
    splits = json.loads(SPLITS_FILE.read_text(encoding="utf-8"))["splits"]
    generated, cleaning = clean_generated(read_jsonl(GENERATION_OUTPUT), fact_base, splits)
    unanswerable = read_jsonl(UNANSWERABLE_FILE)
    if any(row["split"] == "train" for row in unanswerable):
        print("unanswerable queries must not be assigned to train", file=sys.stderr)
        return 1
    by_split = assign_ids(generated + unanswerable)
    report = {"cleaning": cleaning, "splits": {}}
    for split, rows in by_split.items():
        write_jsonl(rows, DATA_DIR / f"{split}.jsonl")
        report["splits"][split] = {
            "queries": len(rows),
            "facts": len({fact_id for row in rows for fact_id in row["relevant_fact_ids"]}),
            "by_bucket": dict(sorted(Counter(row["bucket"] for row in rows).items())),
            "by_language": dict(sorted(Counter(row["language"] for row in rows).items())),
            "long": sum(1 for row in rows if "long" in row["tags"]),
            "numeric": sum(1 for row in rows if "numeric" in row["tags"]),
            "cross_lingual": sum(1 for row in rows if "cross_lingual" in row["tags"]),
        }
    train_facts = {fact_id for row in by_split["train"] for fact_id in row["relevant_fact_ids"]}
    held_out_facts = {fact_id for split in ("validation", "test") for row in by_split[split] for fact_id in row["relevant_fact_ids"]}
    report["fact_overlap_train_vs_heldout"] = len(train_facts & held_out_facts)
    DATASET_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["fact_overlap_train_vs_heldout"] == 0 else 1


def mine_negatives(facts_dir: Path, corpus_dir: Path) -> int:
    import yaml

    from rag.config import EmbeddingConfig
    from rag.dataset_builder import embedding_rows, held_out_chunk_ids, mine_hard_negatives
    from rag.evaluation.relevance import chunk_fact_labels
    from rag.ingestion import CHUNKS_FILE_NAME, read_chunks
    from rag.retrieval.dense import Embedder, passage_text

    config = load_serving_config()
    training = yaml.safe_load((REPO_ROOT / "configs" / "training_config.yaml").read_text(encoding="utf-8"))
    embedding = training["embedding"]
    if embedding["query_instruction"] != config.embedding.query_instruction:
        print("query_instruction differs between training_config.yaml and serving_config.yaml", file=sys.stderr)
        return 1
    fact_base = FactBase.load(facts_dir)
    splits = json.loads(SPLITS_FILE.read_text(encoding="utf-8"))["splits"]
    chunks = read_chunks(config.paths.corpus_processed / CHUNKS_FILE_NAME)
    labels = chunk_fact_labels(chunks, corpus_dir / MANIFEST_NAME)
    passages = {chunk.chunk_id: passage_text(chunk) for chunk in chunks}
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    train = read_jsonl(DATA_DIR / "train.jsonl")
    validation = [row for row in read_jsonl(DATA_DIR / "validation.jsonl") if row["relevant_fact_ids"]]

    embedder = Embedder(config.embedding.model_copy(update={"model": embedding["base_model"]}))
    chunk_vectors = embedder.encode_passages(chunks)
    query_vectors = embedder.encode_queries([row["query"] for row in train])
    excluded = held_out_chunk_ids(labels, splits)
    mined = mine_hard_negatives(
        train, chunk_ids, labels, query_vectors, chunk_vectors, fact_base, excluded, training["data"]["mining_top_k"]
    )
    write_jsonl(mined, DATA_DIR / "hard_negatives.jsonl")

    component_of = {
        fact_id: index for index, component in enumerate(fact_components(read_jsonl(corpus_dir / MANIFEST_NAME))) for fact_id in component
    }
    rows = embedding_rows(
        train, mined, passages, component_of, embedding["negatives_per_row"], embedding["max_positives_per_query"], embedding["seed"]
    )
    write_jsonl(rows, DATA_DIR / "training" / "embedding_train.jsonl")

    from rag.normalize import normalize_for_dense

    ir_validation = {
        "queries": {row["id"]: normalize_for_dense(row["query"]) for row in validation},
        "corpus": passages,
        "relevant": {
            row["id"]: sorted(chunk_id for chunk_id, facts in labels.items() if facts & set(row["relevant_fact_ids"])) for row in validation
        },
    }
    (DATA_DIR / "training" / "ir_validation.json").write_text(json.dumps(ir_validation, ensure_ascii=False), encoding="utf-8")
    confusable_share = sum(1 for record in mined if any(negative["confusable"] for negative in record["negatives"][:5])) / len(mined)
    print(
        json.dumps(
            {
                "train_queries": len(train),
                "training_rows": len(rows),
                "excluded_held_out_chunks": len(excluded),
                "queries_with_confusable_in_top5_negatives": round(confusable_share, 3),
                "validation_queries": len(validation),
                "corpus_chunks": len(passages),
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the knowledge bank corpus and datasets.")
    parser.add_argument("--facts-dir", type=Path, default=None)
    parser.add_argument("--templates-dir", type=Path, default=None)
    parser.add_argument("--corpus-dir", type=Path, default=None)
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("validate-facts", help="Validate fact files and print coverage statistics.")
    subcommands.add_parser("render-kb", help="Render document templates into the raw knowledge bank corpus.")
    subcommands.add_parser("split-facts", help="Assign fact groups to train, validation and test and write the generation input.")
    subcommands.add_parser("build-datasets", help="Clean generated queries and write train, validation and test files.")
    subcommands.add_parser("mine-negatives", help="Mine hard negatives with the base embedding model and write training files.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = load_serving_config().paths
    facts_dir = args.facts_dir or paths.facts_dir
    corpus_dir = args.corpus_dir or paths.corpus_raw
    if args.command == "validate-facts":
        return validate_facts(facts_dir)
    if args.command == "render-kb":
        return render_kb(facts_dir, args.templates_dir or paths.templates_dir, corpus_dir)
    if args.command == "split-facts":
        return split_facts(facts_dir, corpus_dir)
    if args.command == "build-datasets":
        return build_datasets(facts_dir)
    if args.command == "mine-negatives":
        return mine_negatives(facts_dir, corpus_dir)
    return 1


if __name__ == "__main__":
    sys.exit(main())
