# Bilingual Knowledge Bank RAG (Arabic / English)

Self-hosted retrieval-augmented search over an enterprise knowledge bank written in Modern Standard Arabic, English and mixed Arabic/English. The system exposes three modes through one API: **Quick Search** (BM25), **Smart AI Search** (hybrid retrieval, reranking and grounded generation with citations) and **Interactive AI Search** (multi-turn with session memory).

All components are open source and self-hosted. The knowledge bank belongs to a fictional Saudi home-furnishing company, **Qimam Home (قمم للمنزل)**. All names, prices, phone numbers and policies in it are invented.

## Language coverage

| Supported | Notes |
|---|---|
| Modern Standard Arabic | Queries and documents |
| English | Queries and documents |
| Mixed Arabic/English | Arabic text with English terms, product names and codes |
| Cross-lingual | Arabic question → English-only fact, and English question → Arabic-only fact |
| Arabic-Indic digits (٠-٩) | Normalized to 0-9 for search; original text is kept for display |
| Saudi dialect queries | Included in the datasets as a separate bucket (optional assessment bonus) |

## Architecture

```
Knowledge bank (PDF / DOCX / HTML / TXT)
  → parsing (right-to-left aware PDF extraction) → normalization → section-aware chunking → metadata
  → OpenSearch BM25 (bilingual analyzer)        → Quick Search
  → Qwen3-Embedding-0.6B (fine-tuned) → Qdrant  ┐
  → weighted reciprocal rank fusion  ←──────────┘
  → bge-reranker-v2-m3 → calibrated abstention gate
  → Qwen3-4B-Instruct-2507-FP8 on vLLM → grounded answer → deterministic citations
```

## Repository layout

| Path | Purpose |
|---|---|
| `data/facts/` | Controlled ground truth: hand-written facts and entity tables expanded into facts |
| `data/templates/` | Document templates rendered from the facts |
| `data/corpus/raw/` | Rendered knowledge bank (54 documents) and `manifest.jsonl` (section → fact labels) |
| `data/corpus/processed/chunks.jsonl` | Chunks exactly as indexed |
| `data/train.jsonl`, `validation.jsonl`, `test.jsonl` | Retrieval and answer evaluation sets |
| `data/hard_negatives.jsonl` | Mined negatives for training queries |
| `src/rag/` | Library code (parsers, normalization, chunking, retrieval, evaluation) |
| `scripts/` | Entry points required by the assessment |
| `configs/serving_config.yaml`, `configs/training_config.yaml` | Runtime and training configuration |
| `results/` | Measured outputs only |

## Setup

### Development container (CPU)
The repository includes a dev container (`.devcontainer/`) with Python 3.11, Docker, the Arabic fonts and the libraries WeasyPrint needs for PDF rendering.

```bash
docker compose up -d
python -m pytest -q
```

### Manual setup
Python 3.11 is required.

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e . --no-deps
docker compose up -d
```

`docker compose up -d` starts OpenSearch 2.19.1, Qdrant 1.14.1 and Valkey 8.1.

## Building the knowledge bank and indexes

```bash
python scripts/prepare_data.py validate-facts
python scripts/prepare_data.py render-kb
python scripts/build_index.py
python scripts/ingest.py --path data/corpus/raw/
```

`build_index.py --lexical-only` builds only the BM25 index, which is all Quick Search needs on a CPU-only machine.

## Datasets

1. `prepare_data.py split-facts` groups facts that share a document section and assigns whole groups to train, validation or test (seed 2026). No fact and no positive chunk appears in more than one split.
2. `scripts/generate_queries.py` writes six questions per fact with **Qwen3-8B** served by vLLM inside a Hugging Face Job: MSA, long MSA, English, long English, Saudi dialect and code-switched Arabic. The run is seeded (seed 13) and uses JSON-constrained decoding.
3. `prepare_data.py build-datasets` filters the generated questions:
   - wrong script;
   - length outside bounds;
   - the question contains the answer's number;
   - duplicates.

   It then adds hand-written unanswerable questions (validation and test only).
4. `prepare_data.py mine-negatives` mines hard negatives with the base embedding model. Negatives come only from train-split chunks, and no chunk that states the query's own fact is ever used as a negative.

## Running

```bash
python scripts/infer.py --mode quick_search --query "سياسة الاسترجاع"
```

## Results

Results are reported only once they have been measured. They appear in `results/` and are summarized here as each stage completes.
