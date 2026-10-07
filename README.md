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
python scripts/infer.py --mode smart_search --query "كم مدة ضمان المطابخ؟"
```

Smart AI Search needs a GPU, the indexes, and the generator served by vLLM at `generation.url`:

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507-FP8 --gpu-memory-utilization 0.55 --max-model-len 8192 --port 8000
```

## Retrieval and reranking experiments

| Step | Command | Output |
|---|---|---|
| Export BM25, base-dense and fine-tuned-dense rankings with scores | `python scripts/export_rankings.py` | `results/rankings/` |
| Tune weighted RRF on validation and compare retrievers | `python scripts/tune_fusion.py` | `results/fusion/` |
| Build reranker training rows and candidate pools | `python scripts/prepare_data.py build-reranker-data` | `data/training/reranker_*.jsonl` |
| Score with the base reranker, fine-tune, score again (HF Job, 1 GPU) | `python scripts/train_reranker.py --hub-dataset halarash/qimam-kb-rag-data --push` | `results/reranker/scores_*.json` |
| Evaluate rerankers per pool, select on validation, calibrate abstention | `python scripts/evaluate_reranker.py` | `results/reranker/` |
| End-to-end Smart AI Search on the L4 | `scripts/run_l4_smart_search.sh` (HF Job `l4x1`) | `results/smart_search/` |

Every selection (fusion weights, reranker, candidate pool, abstention threshold) uses the validation split only.

## Results

Results are reported only once they have been measured. They appear in `results/` and are summarized here as each stage completes.

### Retrieval (238 answerable test queries; selections made on 153 validation queries)

| System | Recall@1 | Recall@5 | MRR | nDCG@10 | Cross-lingual MRR | AR→EN MRR |
|---|---|---|---|---|---|---|
| BM25 | 0.487 | 0.733 | 0.666 | 0.664 | 0.285 | 0.130 |
| Dense, base Qwen3-Embedding-0.6B | 0.689 | 0.954 | 0.847 | 0.871 | 0.651 | 0.500 |
| Dense, fine-tuned | 0.880 | 0.996 | 0.971 | 0.976 | 0.944 | 0.872 |
| Hybrid, equal weights (RRF k=60) | 0.565 | 0.802 | 0.750 | 0.755 | 0.447 | 0.231 |
| Hybrid, tuned on validation (BM25 weight 0 → dense only) | 0.880 | 0.996 | 0.971 | 0.976 | 0.944 | 0.872 |
| Fine-tuned dense + base reranker | 0.868 | 0.979 | 0.959 | 0.965 | 0.917 | 0.854 |
| **Equal hybrid + fine-tuned reranker (selected)** | **0.901** | **1.000** | **0.986** | **0.990** | **0.975** | **0.935** |

**Fusion.** The validation grid covered BM25 weights 0–1 (dense weight 1) and RRF k ∈ {5, 10, 20, 30, 60, 100}.
- Every non-zero BM25 weight lowered validation nDCG@10, and the loss grew with the weight. The tuned fusion is therefore dense only.
- BM25 is weak on cross-lingual queries (test AR→EN MRR 0.130). With equal weights it pulls the fine-tuned dense ranking down.

**Reranking.**
- The base `bge-reranker-v2-m3` lowers quality after the fine-tuned embedder.
- The fine-tuned reranker raises it on every pool.
- After fine-tuned reranking, the equal-hybrid pool and the dense-only pool are within 0.0004 validation nDCG@10. The hybrid pool was kept by the pre-declared rule: BM25 supplies lexical candidates and the reranker decides their order.

Full per-bucket tables are in `results/fusion/` and `results/reranker/`.

### Smart AI Search, end to end on 1 × NVIDIA L4

Run `day3_l4_20261007-1513` (HF Job `l4x1`, vLLM 0.10.1.1). Pipeline:

1. Equal-weight RRF over BM25 top 50 and fine-tuned dense top 50 gives 30 candidates.
2. The fine-tuned reranker reorders them, and a top-score gate can abstain at this point.
3. Up to 4 passages with reranker score ≥ 0.5 go to Qwen3-4B-Instruct-2507-FP8.
4. Citations are mapped to chunk metadata, then number/code/language post-checks run.

Queries were run sequentially, one at a time. Records and summaries are in `results/smart_search/day3_l4_20261007-1513/`.

**Abstention calibration (validation only).**
- Validation was first run with no gate. The gate threshold was then chosen to maximise the balanced accuracy of the final answered/NOT_FOUND decision.
- Without the gate, the generator alone returned NOT_FOUND for 88.5% of unanswerable validation questions and answered 97.4% of answerable ones.
- With the gate, validation reaches 100% NOT_FOUND on unanswerable questions while answering 95.4% of answerable ones.

| Test split (238 answerable, 34 unanswerable) | Value |
|---|---|
| Answerable questions answered | 0.979 (233 / 238) |
| Unanswerable questions → NOT_FOUND | 1.000 (34 / 34) |
| Relevant chunk in the generator's context | 0.971 |
| Answers whose citations include a relevant chunk | 0.987 |
| Citation precision (cited chunks that are relevant) | 0.977 |
| Answers with no valid citation marker (fallback used) | 0.000 |
| Answer in the expected language | 0.996 |
| Reference numbers all present in the answer (product codes excluded) | 0.891 (n = 175) |
| Answer numbers all present in the reference | 0.977 (n = 171) |
| Answers rejected by the number/code post-check | 0 |

Latency in ms (all 272 test queries; queries stopped by the gate skip generation):

| Stage | avg | p50 | p95 | max |
|---|---|---|---|---|
| BM25 | 5.3 | 5.3 | 6.4 | 15.4 |
| Query embedding | 34.8 | 34.3 | 38.5 | 45.3 |
| Dense search | 4.8 | 4.7 | 5.4 | 6.4 |
| Rerank (30 pairs) | 43.9 | 43.2 | 57.0 | 66.3 |
| Generation (237 queries) | 698 | 690 | 1,099 | 1,454 |
| **Total** | **698** | **743** | **1,170** | **1,567** |

**GPU memory.** Peak use was 16.2 GB of 23.0 GB:
- vLLM reservation: 13.3 GB (`--gpu-memory-utilization 0.55`).
- Embedder and reranker: 2.3 GB peak allocated.

**Caveats.**
- These are single-user latencies. Concurrency, corpus scaling and the full generation-quality evaluation are Day 5 work.
- The test run used the threshold 0.8887. Calibration now picks midpoints between observed scores (0.8445), which fixes a float-equality edge case seen on validation. One unanswerable test query (score 0.883) lies between the two thresholds, and its outcome under 0.8445 has not been measured yet.
