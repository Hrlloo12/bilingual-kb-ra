# Bilingual Knowledge Bank RAG (Arabic / English)

Self-hosted retrieval-augmented search over an enterprise knowledge bank written in Modern Standard Arabic, English and mixed Arabic/English. One API serves three modes:

| Mode | What it does |
|---|---|
| **Quick Search** (`quick_search`) | BM25 keyword search with Arabic and English normalization; no LLM; runs on CPU |
| **Smart AI Search** (`smart_ai_search`) | Hybrid retrieval with a fine-tuned embedder, a fine-tuned reranker and a local LLM. Returns a grounded answer with citations, or NOT_FOUND |
| **Interactive AI Search** (`interactive`) | Multi-turn chat with session memory, follow-up rewriting and suggested follow-up questions |

Every model and service is open source and self-hosted. The serving stack (OpenSearch, Qdrant, Valkey, embedder, reranker, vLLM generator and API) runs on one NVIDIA L4 24 GB.

The knowledge bank belongs to a fictional Saudi home-furnishing company, **Qimam Home (قمم للمنزل)**. All names, prices, phone numbers and policies are invented.

## Contents

1. [Language coverage](#language-coverage)
2. [Architecture](#architecture)
3. [Models and licenses](#models-and-licenses)
4. [Dataset](#dataset)
5. [Fine-tuning](#fine-tuning)
6. [Repository layout](#repository-layout)
7. [Reproducing the project](#reproducing-the-project)
8. [Results](#results)
9. [Optimizations tried](#optimizations-tried)
10. [Deployment notes](#deployment-notes)
11. [Known limitations](#known-limitations)
12. [Possible improvements](#possible-improvements)

## Language coverage

| Supported | Notes |
|---|---|
| Modern Standard Arabic | Questions and documents |
| English | Questions and documents |
| Mixed Arabic/English | Arabic text with English terms, product names and codes, in questions and documents |
| Cross-lingual | Arabic question → fact that exists only in English documents, and the reverse |
| Arabic-Indic digits (٠–٩) | Normalized to 0–9 for search; original text kept for display |
| Saudi dialect questions | Optional bonus; a separate bucket in every dataset split and evaluation |

Answers follow the language of the question. Mixed questions are answered in Arabic, with English product names and terms kept as written.

## Architecture

```
                        PDF / DOCX / HTML / TXT documents
                                      │
             parse (RTL-aware PDF) → normalize → section-aware chunks + metadata
                     │                                         │
          OpenSearch BM25 (bilingual analyzer)     Qwen3-Embedding-0.6B (fine-tuned) → Qdrant
                     │                                         │
 Quick Search ◄──────┤                                         │
                     └──────── weighted RRF (equal weights, 30 candidates) ◄──┘
                                      │
                         bge-reranker-v2-m3 (fine-tuned)
                                      │
                    NOT_FOUND gate (top score < 0.8445, calibrated on validation)
                                      │
                Qwen3-4B-Instruct-2507-FP8 on vLLM (streaming, temperature 0)
                                      │
     citations mapped from chunk metadata → number/code/language post-checks → Smart AI Search
                                      │
  Valkey session memory → rule-based follow-up gate → LLM rewrite → same pipeline → Interactive
                                      └→ suggested follow-up questions (generated in parallel)
```

**Why this design for Arabic and English.**
- **Keyword search.** BM25 with Arabic normalization handles exact codes, names and numbers, and serves Quick Search without a GPU. On its own it cannot cross languages: test MRR is 0.130 for Arabic questions about English-only facts.
- **Dense retrieval.** A multilingual embedder bridges Arabic and English. Fine-tuning it on this knowledge bank raised cross-lingual MRR from 0.651 to 0.944.
- **Reranking.** A multilingual cross-encoder reorders the fused candidates. Its top score is also the confidence signal behind the NOT_FOUND gate.
- **Generator.** A 4B instruct model in FP8 answers in Arabic and English, fits next to the retrieval models on one L4, and keeps generation under one second on average.
- **Citations.** Every citation field comes from chunk metadata; the model only picks passage numbers. Citations therefore cannot point to documents that were not retrieved.

## Models and licenses

| Component | Model | License (commercial use) | Size on disk | GPU memory | CPU | GPU | Quantization |
|---|---|---|---|---|---|---|---|
| Embedder (fine-tuned) | [halarash/qimam-qwen3-embedding-0.6b-kb-v2](https://huggingface.co/halarash/qimam-qwen3-embedding-0.6b-kb-v2), from [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | Apache-2.0 (yes) | 2.4 GB fp32 | 2.3 GB peak together with the reranker, fp16 | Yes | Yes | None (fp16 on GPU) |
| Reranker (fine-tuned) | [halarash/qimam-bge-reranker-v2-m3-kb](https://huggingface.co/halarash/qimam-bge-reranker-v2-m3-kb), from [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | Apache-2.0 (yes) | 2.27 GB fp32 | (included above) | Yes | Yes | None (fp16 on GPU) |
| Generator | [Qwen/Qwen3-4B-Instruct-2507-FP8](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507-FP8) | Apache-2.0 (yes) | 4.2 GiB of weights | 13.3 GB vLLM reservation (4.2 GiB weights + KV cache) | Not used | Yes | FP8 (official block-wise FP8) |
| Evaluation judge (offline only) | [Qwen/Qwen3-8B-FP8](https://huggingface.co/Qwen/Qwen3-8B-FP8) | Apache-2.0 (yes) | 9.5 GB | 17.3 GB while judging, serving stack stopped | – | Yes | FP8 |
| Question generation (dataset only) | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Apache-2.0 (yes) | 16 GB | – | – | Yes | None |

The generator and the rewrite/follow-up prompts share the same vLLM model.

| Infrastructure and libraries | License |
|---|---|
| OpenSearch 2.19.1, Qdrant 1.14.1, vLLM 0.10.1.1 | Apache-2.0 |
| Valkey 8.1 | BSD-3-Clause |
| PyTorch, lxml, httpx, uvicorn, psutil, NumPy, Jinja2, WeasyPrint | BSD-3-Clause |
| transformers, sentence-transformers, huggingface_hub, datasets, accelerate, opensearch-py, qdrant-client, sacrebleu | Apache-2.0 |
| FastAPI, pydantic, PyYAML, python-docx, beautifulsoup4, valkey-py | MIT |
| **PyMuPDF** (PDF parsing) | **AGPL-3.0 or Artifex commercial license** |

PyMuPDF is the only copyleft dependency. Offering the system as a network service under AGPL-3.0 requires publishing the corresponding source. A closed commercial deployment needs a commercial PyMuPDF license, or a permissive parser such as pypdf (BSD-3-Clause) with the Arabic right-to-left fix in `src/rag/parsers/pdf_parser.py` re-validated.

The full model card is [`model_card.md`](model_card.md). Upload-ready Hugging Face cards are in [`hub_cards/`](hub_cards/).

## Dataset

Hugging Face: [`halarash/qimam-kb-rag-data`](https://huggingface.co/datasets/halarash/qimam-kb-rag-data). The same data is in `data/`. License: CC-BY-4.0, synthetic. The card is [`hub_cards/dataset/README.md`](hub_cards/dataset/README.md).

| Part | Size |
|---|---|
| Ground-truth facts (`data/facts/`) | 299 facts in 14 domains |
| Knowledge bank (`data/corpus/raw/`) | 54 documents: 20 Arabic, 18 English, 16 mixed; 15 PDF, 14 DOCX, 16 HTML, 9 TXT; 6 with Eastern Arabic digits |
| Indexed chunks (`data/corpus/processed/chunks.jsonl`) | 292 |
| Query → relevant chunk pairs (`data/train.jsonl`, `validation.jsonl`, `test.jsonl`) | 1,277 / 179 / 272 questions, split by fact group so no fact is shared across splits |
| Question → reference answer (same files, `reference_answer`) | Every validation and test question, including 26 / 34 unanswerable ones (reference `NOT_FOUND`) |
| Hard negatives (`data/hard_negatives.jsonl`) | Mined for every training question |
| Multi-turn traces (`data/interactive/`) | 227 validation and 346 test two-turn conversations with gold standalone rewrites |

**How the data was made.**
1. Facts were written by hand, and documents were rendered from them in four formats and three language styles.
2. Qwen3-8B wrote six questions per fact: MSA, long MSA, English, long English, Saudi dialect and code-switched Arabic.
3. Automatic filters removed questions with the wrong script, wrong length, the answer embedded in the question, and duplicates.
4. Every test question was read by hand; 17 were corrected (`data/review/test_review.yaml`).
5. Unanswerable questions were written by hand.

Known limitations: the data is synthetic and small, the questions share one generator's style, and only the test split was reviewed by hand.

## Fine-tuning

Both retrieval models were fully fine-tuned (all layers). The generator was not fine-tuned; grounding comes from retrieval, the prompt, deterministic citations and post-checks. Hyperparameters are in `configs/training_config.yaml`.

| | Embedder | Reranker |
|---|---|---|
| Base | Qwen/Qwen3-Embedding-0.6B | BAAI/bge-reranker-v2-m3 |
| Why | Multilingual, small (0.6B), instruction-aware, Apache-2.0 | Multilingual cross-encoder, Apache-2.0 |
| Method | Full fine-tuning, sentence-transformers | Full fine-tuning, `CrossEncoderTrainer` |
| Loss | `CachedMultipleNegativesRankingLoss` (InfoNCE, in-batch + 2 mined hard negatives) | `BinaryCrossEntropyLoss`, `pos_weight` 4 |
| Rows | 1,438 from 1,277 training questions | 1,438 (1 positive : 4 negatives), 7,190 pairs |
| Hard negatives | Mined with the base embedder; never a chunk stating the query's own fact (so never its translation); confusable facts first | 3 from the embedder's hard negatives, 1 from BM25 top results |
| Batch / epochs | 64 (cache mini-batch 16) / 3, best epoch selected on validation nDCG@10 | 32 / 1 |
| LR / warmup / weight decay | 1e-5 / 0.1 / 0.01, bf16 | 1e-5 / 0.1 / 0.01, bf16 |
| Hardware / time | 1 × A10G, 453 s | 1 × A10G, 72.5 s |

**Shared details.**
- **Split:** train / validation / test by fact group, seed 2026. All model, fusion, reranker and threshold choices used validation only.
- **Normalization:** NFKC; bidirectional control characters, tashkeel and tatweel removed; Arabic-Indic digits converted to ASCII. BM25 uses a custom bilingual OpenSearch analyzer: Arabic normalization (alef, ya and ta-marbuta variants), Arabic light stemming, Arabic and English stop words, and English possessive and Porter-style stemming. Passages are prefixed with `title | section`.
- **Augmentation:** each fact is paraphrased in six styles, and cross-lingual pairs come from facts that exist in only one language. Back-translation was not used.
- **Batching:** a group-aware batch sampler keeps the Arabic and English chunks of the same fact out of each other's in-batch negatives.
- **Latency after fine-tuning:** both models serve in fp16 on GPU. The same architecture means the same speed as the base models (see the before/after table).

## Repository layout

```
├── README.md, model_card.md, requirements.txt, pyproject.toml, docker-compose.yml
├── configs/            serving_config.yaml, training_config.yaml
├── data/               facts/, templates/, corpus/ (raw documents + processed chunks), train/validation/test.jsonl,
│                       hard_negatives.jsonl, training/, interactive/, generation/, review/
├── scripts/            prepare_data.py, build_index.py, train_embeddings.py, train_reranker.py, ingest.py, infer.py,
│                       benchmark_retrieval.py, benchmark_generation.py, benchmark_latency.py, benchmark_concurrency.py,
│                       benchmark_scaling.py, evaluate_smart_search.py, evaluate_interactive.py, build_reports.py, ...
├── src/rag/            library: parsers, normalization, chunking, retrieval, reranker, generation, interactive, API
├── demo/               app.py (scripted live demo), demo_queries.json, web/ (UI served by the API), sample_docs/
├── docker/             api.Dockerfile
├── hub_cards/          Hugging Face cards: embedding/, reranker/, dataset/
├── results/            before/after metrics, reports, error analysis, human evaluation, raw run outputs
└── tests/              157 unit tests
```

## Reproducing the project

Commands are labelled by where they need to run: **CPU** (any machine), **GPU** (one NVIDIA GPU with at least 24 GB, the L4 for the official numbers), or **HF Job** (a Hugging Face GPU job, which is how the training runs were made).

### 1. Environment (CPU)

Python 3.11. WeasyPrint needs Pango and the Noto fonts (see `.devcontainer/setup.sh`) only to re-render the PDF documents.

```bash
git clone https://github.com/Hrlloo12/bilingual-kb-ra.git && cd bilingual-kb-ra
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu   # or the CUDA wheel on a GPU host
pip install -r requirements.txt
pip install -e . --no-deps
docker compose up -d          # OpenSearch 2.19.1, Qdrant 1.14.1, Valkey 8.1
python -m pytest -q           # 157 tests
```

The fine-tuned models are private repositories. Set `HF_TOKEN` to a token that can read them (`export HF_TOKEN=...`, or `HF_TOKEN=...` in `.env` for Compose).

### 2. Data (CPU, except question generation)

```bash
python scripts/prepare_data.py validate-facts         # check the facts
python scripts/prepare_data.py render-kb              # facts + templates → data/corpus/raw (54 documents)
python scripts/prepare_data.py split-facts            # fact groups → train/validation/test, writes generation input
python scripts/generate_queries.py --repo-id halarash/qimam-kb-rag-data   # HF Job / GPU: Qwen3-8B on vLLM writes questions
python scripts/prepare_data.py build-datasets         # filter questions, add unanswerable ones
python scripts/prepare_data.py mine-negatives         # hard negatives with the base embedder
python scripts/prepare_data.py build-reranker-data    # reranker rows and candidate pools
python scripts/build_conversations.py                 # interactive conversations
```

All of these outputs are committed, so this step can be skipped.

### 3. Fine-tuning (HF Job or GPU)

```bash
python scripts/train_embeddings.py --hub-dataset halarash/qimam-kb-rag-data
python scripts/train_reranker.py --hub-dataset halarash/qimam-kb-rag-data
```

The models are saved under `artifacts/embedding` and `artifacts/reranker`. `--push` also uploads them to the repositories named in `configs/training_config.yaml`, and `--hub-model-id` overrides the embedder destination. Point `embedding.model` and `reranker.model` in `configs/serving_config.yaml` (or the `EMBEDDING_MODEL` and `RERANKER_MODEL` environment variables) at the new models.

### 4. Indexes and ingestion (CPU for BM25; GPU recommended for embeddings)

```bash
python scripts/build_index.py                       # BM25 index + Qdrant collection from data/corpus/raw
python scripts/build_index.py --lexical-only        # BM25 only: everything Quick Search needs
python scripts/ingest.py --path demo/sample_docs/   # add or replace PDF/DOCX/TXT/HTML documents in both indexes
```

### 5. Inference, one command per mode

Smart AI Search and Interactive need the generator:

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507-FP8 --gpu-memory-utilization 0.55 --max-model-len 8192 --port 8000   # GPU
```

```bash
python scripts/infer.py --mode quick_search --query "سياسة الاسترجاع"                                   # CPU
python scripts/infer.py --mode smart_ai_search --query "What is the SLA for premium support tickets?"     # GPU
SESSION=$(python scripts/infer.py --mode interactive --session new --query "Where is the Abha showroom?" | python -c "import json, sys; print(json.load(sys.stdin)['session_id'])")
python scripts/infer.py --mode interactive --session "$SESSION" --query "وش رقم الـ phone حقه؟"         # GPU; needs Valkey
```

### 6. API, UI and demo

```bash
docker compose --profile gpu up -d --build    # GPU host: full stack, API and UI on http://localhost:8080
docker compose --profile cpu up -d --build    # CPU-only host: Quick Search API and UI on http://localhost:8080
```

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/search` | `{"mode": "quick_search" \| "smart_ai_search" \| "interactive", "query": "...", "session_id": "...", "top_k": n}`; `smart_search` is accepted as an alias |
| `GET` / `DELETE` | `/v1/sessions/{id}` | Read or end a conversation (turns, TTL) |
| `GET` | `/health`, `/livez` | Readiness of every component, and liveness |
| `GET` | `/` | Web UI: pick a mode, ask in Arabic or English, see answers, citations, latency and suggested follow-ups |

Responses follow the assessment payloads:
- `citations[]`: `id`, `doc_id`, `chunk_id`, `title`, `section`, `snippet`, `page`, `source`, `relevance`.
- Smart AI Search `latency_ms`: `preprocessing`, `retrieval`, `rerank`, `time_to_first_token`, `generation_total`, `post_checks`, `total`.
- Interactive adds `rewritten_query`, `suggested_followups` and `query_rewrite` latency.

The scripted live demo plays the demo questions against a running API, then prints the measured tables:

```bash
python demo/app.py run --base-url http://localhost:8080 --pause
python scripts/ingest.py --path demo/sample_docs/
python demo/app.py run --set after_ingestion
python demo/app.py results
```

`demo/demo_queries.json` lists the questions:
- Quick Search in Arabic and English;
- the assessment's own examples;
- long questions, and questions with prices and numbers;
- cross-lingual questions in both directions;
- mixed questions and a NOT_FOUND case;
- two code-switching multi-turn conversations;
- two questions answered only by the newly ingested documents.

A transcript of these steps on the final L4 stack is in `results/demo/`. `demo_run.json`, `ingest.json` and `demo_after_ingestion.json` hold every response with its citations and latency. All answerable demo questions were answered with citations. The documented pricing example returned NOT_FOUND, as described under Known limitations.

### 7. Benchmarks

The official numbers come from one run on an NVIDIA L4 24 GB: `TASK=final scripts/run_l4_job.sh`. It starts the same services as Compose, as processes, then runs everything below. The individual commands, against a running stack:

```bash
python scripts/benchmark_retrieval.py --retrievers bm25 dense hybrid hybrid_rerank     # retrieval quality per bucket and language
python scripts/export_rankings.py && python scripts/tune_fusion.py && python scripts/evaluate_reranker.py   # fusion and reranker selection (validation)
python scripts/evaluate_smart_search.py --splits validation test --label final          # end-to-end answers and latency (GPU)
python scripts/benchmark_generation.py --records results/smart_search/final_test.jsonl --judge-url http://127.0.0.1:8001/v1   # needs the judge on vLLM
python scripts/evaluate_interactive.py --splits test                                    # multi-turn retrieval and answers through the API
python scripts/benchmark_latency.py api --output results/latency_api.json               # per-mode, per-stage latency through the API
python scripts/benchmark_latency.py topk --output results/latency_topk.json             # latency against retrieved/reranked top-k (GPU)
python scripts/benchmark_concurrency.py --mode smart_ai_search --output results/concurrency_smart.json   # also quick_search, interactive
python scripts/benchmark_scaling.py --output results/scaling.json                       # 1k / 10k / 100k chunks (GPU)
python scripts/build_human_eval.py --records results/smart_search/final_test.jsonl      # 30-item human evaluation sheet
python scripts/summarize_human_eval.py && python scripts/build_reports.py               # reports and before/after table
```

The before-fine-tuning stack is the same pipeline with environment overrides:

```bash
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B QDRANT_COLLECTION=kb_chunks_base RERANKER_MODEL=BAAI/bge-reranker-v2-m3 <command>
```

Its NOT_FOUND threshold is re-calibrated on validation with `scripts/calibrate_abstention.py`.

## Results

All numbers are measured. The raw outputs of the final run are in `results/final_l4_20261008/`. The reports are generated from them by `scripts/build_reports.py`:

- [`results/before_vs_after.md`](results/before_vs_after.md)
- [`results/retrieval_report.md`](results/retrieval_report.md)
- [`results/generation_report.md`](results/generation_report.md)
- [`results/latency_report.md`](results/latency_report.md)
- [`results/concurrency_report.md`](results/concurrency_report.md)
- [`results/error_analysis.md`](results/error_analysis.md)
- [`results/before_finetuning_metrics.json`](results/before_finetuning_metrics.json) and [`results/after_finetuning_metrics.json`](results/after_finetuning_metrics.json)

**Hardware.** GPU results: one NVIDIA L4 24 GB (23,034 MiB usable, driver 595.71.05) with an Intel(R) Xeon(R) Platinum 8268 CPU @ 2.90GHz (24 vCPUs, 251 GiB RAM), Vast.ai container, vLLM 0.10.1.1, run `final_l4_20261008` on 2026-10-08. CPU-only Quick Search: a 4-vCPU AMD EPYC 7763 64-Core Processor with 15 GiB RAM and no GPU (GitHub Codespace).

### Before vs after fine-tuning

|  | Before | After |
|---|---|---|
| Embedder | Qwen/Qwen3-Embedding-0.6B | halarash/qimam-qwen3-embedding-0.6b-kb-v2 |
| Reranker | BAAI/bge-reranker-v2-m3 | halarash/qimam-bge-reranker-v2-m3-kb |
| NOT_FOUND gate threshold (validation-calibrated) | 0.0104 | 0.8445 |

| Metric | Before fine-tuning | After fine-tuning |
|---|---|---|
| Recall@5 (AR) | 0.985 | 1.000 |
| Recall@5 (EN) | 0.976 | 1.000 |
| Recall@5 (mixed) | 1.000 | 1.000 |
| Recall@5 (cross-lingual) | 0.946 | 1.000 |
| MRR (AR) | 0.939 | 0.976 |
| MRR (EN) | 0.971 | 1.000 |
| nDCG@10 (AR) | 0.949 | 0.982 |
| nDCG@10 (EN) | 0.975 | 1.000 |
| Faithfulness (LLM judge, upper bound) | 0.996 | 0.996 |
| Answer relevance (LLM judge, upper bound) | 0.991 | 0.991 |
| Hallucination rate among answered (LLM judge, lower bound) | 0.004 | 0.004 |
| Answerable questions answered | 0.945 | 0.979 |
| Unanswerable questions → NOT_FOUND | 0.971 | 0.971 |
| Citation precision (facts) | 0.990 | 0.983 |
| Human meaning score (1–5) | not rated | 3.80 |
| Human fluency score, Arabic answers (1–5) | not rated | 4.00 |
| Human fluency score, English answers (1–5) | not rated | 4.00 |
| Time to first token, avg (ms) | 167 | 162 |
| Time to first token, p95 (ms) | 186 | 182 |
| Total latency, Smart AI, avg (ms) | 732 | 736 |
| Total latency, Smart AI, p95 (ms) | 1,217 | 1,229 |
| Quick Search p95 latency (ms, no fine-tuned component) | 25.0 | 25.0 |
| GPU memory used, whole stack (MiB) | 16,158 | 16,230 |
| GPU memory, embedder + reranker peak allocated (MiB) | 2,319 | 2,324 |
| CPU memory, host RAM used (MiB) | 13,361 | 13,743 |
| CPU memory, retrieval process RSS (MiB) | 2,460 | 2,782 |

**What fine-tuning changed.**
- **Retrieval improved.** Cross-lingual Recall@5 of the full stack rose from 0.946 to 1.000, and Arabic MRR from 0.939 to 0.976. Measured dense-only, the fine-tuned embedder raised cross-lingual MRR from 0.651 to 0.944 (retrieval table below).
- **More questions were answered.** Answerable questions answered rose from 0.945 to 0.979: false NOT_FOUNDs fell from 0.055 to 0.021, and context precision rose from 0.727 to 0.929 (generation report). Unanswerable handling stayed at 0.971.
- **Answer correctness was equally high where both stacks answered.** The LLM judge finds one unsupported answer in each run (`test_00242`), so its faithfulness, relevance and hallucination figures cannot separate the stacks.
- **Latency and memory did not change**, as expected: same architectures and sizes. Fine-tuning improved retrieval and answer coverage, not speed.
- **The base reranker's scores sit on a different scale**, so its validation-calibrated NOT_FOUND threshold is 0.0104, not 0.8445.
- **Human scores** cover the final (after) stack only. The same 30 questions answered by the base stack are in `results/human_eval/human_eval_before_sheet.csv` and were not rated.

### Retrieval quality (test split, 238 answerable questions)

| System | R@1 | R@5 | R@10 | R@20 | Hit@1 | Hit@5 | Hit@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|---|---|---|
| BM25 only (Quick Search) | 0.487 | 0.733 | 0.773 | 0.802 | 0.546 | 0.790 | 0.824 | 0.666 | 0.664 |
| Dense only, base embedder | 0.689 | 0.954 | 0.983 | 0.996 | 0.760 | 0.962 | 0.983 | 0.847 | 0.871 |
| Dense only, fine-tuned embedder | 0.880 | 0.996 | 0.996 | 1.000 | 0.954 | 0.996 | 0.996 | 0.971 | 0.976 |
| Hybrid BM25 + fine-tuned dense (equal RRF) | 0.565 | 0.802 | 0.884 | 0.990 | 0.634 | 0.849 | 0.916 | 0.750 | 0.755 |
| Hybrid, weights tuned on validation | 0.880 | 0.996 | 0.996 | 1.000 | 0.954 | 0.996 | 0.996 | 0.971 | 0.976 |
| Hybrid + base reranker | 0.868 | 0.979 | 0.992 | 1.000 | 0.941 | 0.979 | 0.992 | 0.959 | 0.966 |
| Hybrid + fine-tuned reranker (deployed) | 0.901 | 1.000 | 1.000 | 1.000 | 0.975 | 1.000 | 1.000 | 0.986 | 0.990 |

Per-language Recall@1/5/10/20, Hit@k, MRR and nDCG@10 for every system are in [`results/retrieval_report.md`](results/retrieval_report.md).

### Answer quality (Smart AI Search, test split)

| Metric | Before | After |
|---|---|---|
| Faithfulness: all claims supported (judge, upper bound) | 0.996 | 0.996 |
| Hallucination rate among answered (judge, lower bound) | 0.004 | 0.004 |
| Answer relevance (judge, upper bound) | 0.991 | 0.991 |
| Context precision (facts) | 0.727 | 0.929 |
| Context recall (facts) | 0.950 | 0.971 |
| Citation precision (facts) | 0.990 | 0.983 |
| Citation recall (facts) | 0.991 | 0.987 |
| Answer in the expected language | 0.996 | 0.991 |
| Unanswerable → NOT_FOUND | 0.971 | 0.971 |
| False NOT_FOUND on answerable | 0.055 | 0.021 |
| BLEU vs reference | 54.5 | 53.5 |
| chrF vs reference | 66.3 | 65.3 |
| ROUGE-L vs reference | 0.699 | 0.686 |

The LLM-judge figures are bounds: on the errors found by hand, the judge missed 4 of 5 wrong answers. See [`results/error_analysis.md`](results/error_analysis.md).

### Latency, one request at a time

| Mode | Requests | avg | p50 | p95 | max | First result p50 | First result p95 | First token p50 | First token p95 |
|---|---|---|---|---|---|---|---|---|---|
| Quick Search | 272 | 17.7 | 17.1 | 25.0 | 47.2 | 13.1 | 20.2 | – | – |
| Smart AI Search | 100 | 769 | 824 | 1,260 | 1,437 | 78.6 | 92.5 | 176 | 192 |
| Interactive AI Search | 100 | 1,264 | 1,232 | 1,815 | 2,266 | 82.4 | 706 | 222 | 800 |

Targets: Quick Search p95 < 100 ms, Smart AI Search first token p95 < 1,500 ms and total p95 < 4,000 ms, interactive rewrite < 200 ms. All are met except the rewrite: 337 ms p50 and 662 ms p95 on follow-up turns (see Optimizations tried). On the CPU-only host, Quick Search p95 is 15.4 ms.

The full report is [`results/latency_report.md`](results/latency_report.md). It covers:
- per-stage breakdowns for every mode;
- time to first result;
- maximum input length (up to the 1,000-character API limit);
- latency against retrieved and reranked top-k;
- latency against corpus size (1k, 10k and 100k chunks);
- the CPU-only run.

### Concurrency

p95 latency / requests per second / failures. Closed loop: each user sends its next request when the previous one returns. Interactive users hold real two-turn conversations.

| Users | Quick Search (L4 host) | Smart AI Search | Interactive | Quick Search (CPU-only host) |
|---|---|---|---|---|
| 1 | 14 ms / 78.4 / 0 | 1,258 ms / 1.3 / 0 | 1,787 ms / 0.8 / 0 | 10 ms / 128 / 0 |
| 2 | 16 ms / 163.3 / 0 | 1,413 ms / 2.2 / 0 | 1,908 ms / 1.4 / 0 | 16 ms / 184 / 0 |
| 4 | 24 ms / 232.9 / 0 | 1,698 ms / 3.7 / 0 | 2,299 ms / 2.3 / 0 | 24 ms / 230 / 0 |
| 8 | 51 ms / 218.5 / 0 | 2,297 ms / 5.3 / 0 | 2,846 ms / 3.7 / 0 | 57 ms / 227 / 0 |
| 16 | 90 ms / 240.6 / 0 | 3,642 ms / 5.6 / 0 | 4,825 ms / 4.4 / 0 | 112 ms / 233 / 0 |

- **No failures** at any level in any mode.
- **GPU saturation.** Smart AI Search throughput levels off at about 5.6 requests/s from 8 users, with the GPU at 99%. Latency then grows by queueing, and p95 stays under 4 s at 16 users.
- **Resources.** GPU memory stays at about 16.0 GiB, and host RAM at about 13.9 GB.
- **Full report.** Average, p50 and max latency, GPU utilization and memory, CPU and host RAM per level are in [`results/concurrency_report.md`](results/concurrency_report.md).

### Human evaluation

`results/human_eval/human_eval_sheet.csv` holds 30 answers from the final run, rated by a person: 10 Arabic, 10 English, 10 mixed or cross-lingual. The sampling is stratified by question type and includes 3 unanswerable questions. The summary is in `results/human_eval/human_eval_summary.json`.

| Criterion (1–5) | Mean |
|---|---|
| Meaning / factual accuracy | 3.80 |
| Faithfulness to the retrieved context | 3.97 |
| Fluency, Arabic answers | 4.00 |
| Fluency, English answers | 4.00 |
| Completeness | 3.80 |
| Citation quality | 3.83 |
| Answer in the question's language | 30 / 30 |

Two answers were rated 1 for meaning and completeness. Both are known errors listed in [`results/error_analysis.md`](results/error_analysis.md):
- an Arabic question about the Khobar showroom's phone number returned NOT_FOUND although the answer exists in English;
- a WhatsApp customer-service question was answered with a showroom phone number.

The other 28 were rated 4 (good) on every criterion. The rating instructions are in [`results/human_eval/README.md`](results/human_eval/README.md).

### Error analysis

[`results/error_analysis.md`](results/error_analysis.md) gives concrete, hand-checked examples for every category the assessment lists:
- wrong document;
- hallucination;
- citations;
- language drift;
- mixed queries;
- conversation memory;
- numbers, prices, names and negation;
- untranslated text;
- NOT_FOUND handling.

## Optimizations tried

Each optimization is marked by whether it was measured. Numbers are from the final L4 run unless another run is named.

| Technique | Status | Measured impact |
|---|---|---|
| FP8 generator (official Qwen3-4B-Instruct-2507-FP8) on vLLM | Used | 4.2 GiB of weights. The whole stack fits in 16,406 MiB of the L4's 23,034 MiB. Not compared against BF16 |
| Streaming generation (vLLM SSE) | Used inside the server | Gives the exact first-token time: Smart AI Search first token p95 192 ms, against the 1,500 ms target. Tokens are not streamed to the client, because the number/code post-checks need the whole answer before it is released |
| NOT_FOUND gate before generation | Used | Questions stopped by the gate skip the LLM. In the smoke test, the NOT_FOUND answer took 169 ms, against 824 ms p50 for answered questions |
| Rule-based rewrite gate (no model call for standalone questions) | Used | Standalone turns skip the rewrite. The interactive rewrite stage has p50 0 ms over all turns, and its memory read + write averages 0.5 ms (rewrite-selection run) |
| Suggested follow-ups generated in parallel with the answer | Used | They take 930 ms on average, but start as soon as reranking ends. The extra wait after the answer averages 242 ms (p95 791 ms) |
| fp16 embedder and reranker on GPU; 30 pairs reranked in one batch | Used | Query embedding 49 ms and reranking 52 ms at p50 |
| Candidate and context sizes (30 reranked, ≤ 4 passages to the LLM) | Measured | See the top-k sweep: reranking 10 → 50 candidates moves rerank p50 from 25 to 82 ms and total p50 from 762 to 829 ms, while the relevant chunk reaches the generator for 0.874 of answerable questions at 10, 0.966 at 30 and 0.966 at 50. Sending 1 to 8 passages changes total p50 by under 10 ms because prompts stay short (245–257 tokens), and 1 passage loses some context hits (0.954 vs 0.966); 30 candidates and 4 passages were kept |
| Approximate nearest neighbours (Qdrant HNSW) | Used automatically | Qdrant searches exactly below its 20 MB segment threshold (1k and 10k chunks) and uses HNSW at 100k. Dense search p50 stays at 15–17 ms from 1k to 100k chunks (corpus-scaling run) |
| BM25 weight tuned on validation | Tried | The best validation fusion was dense only (BM25 weight 0). The deployed equal-weight hybrid + fine-tuned reranker was within 0.0004 validation nDCG@10 and was kept by the pre-declared rule. BM25 costs 18 ms p50 |
| Embedding quantization, ONNX export, torch.compile | Not tried | Retrieval stages total about 130 ms p50, a small share of end-to-end latency, which generation dominates |
| Speculative decoding, KV-cache reuse across turns, query/rewrite cache | Not tried | vLLM's automatic prefix caching (on by default) reuses the shared system prompt; its effect was not measured separately |

**The < 200 ms query-rewrite target was not achieved.**
- On follow-up turns, the rewrite takes 337 ms at p50 and 662 ms at p95, because it is a full LLM call on the same GPU as answer generation.
- The rule-based gate keeps standalone turns at 0 ms, which is why the interactive turn still meets the Smart AI Search targets (first token p95 800 ms, total p95 1,815 ms).
- Reaching 200 ms would need a smaller dedicated rewrite model, or a rewrite cache.

## Deployment notes

- **GPU stack.** `docker compose --profile gpu up -d --build` was validated from a clean start (no images, volumes or caches) on an RTX 4090 24 GB VM with `scripts/clean_start_test.sh` (`results/deployment/clean_start_20261007T160306Z/`). No L4 VM with Docker was available. The official L4 numbers therefore come from `scripts/run_l4_job.sh`, which runs the same service versions and the same API code as processes.
- **CPU-only Quick Search.** `docker compose --profile cpu up -d --build` was validated on a 4-vCPU CPU-only machine (`results/cpu_quick_search/deployment.txt`). It builds a 2.1 GB image with CPU PyTorch and starts OpenSearch and an API with `API_MODES=quick_search`; the other modes answer 503. `API_MODES` accepts any subset of the three modes.
- **Ports and security.** OpenSearch, Qdrant and Valkey listen on localhost only. The API has no authentication or rate limiting, so reach it through an SSH tunnel that forwards port 8080, or through an authenticated proxy.
- **Secrets.** The HF token is read from `.env`, which git ignores.

## Known limitations

- **Small, synthetic knowledge bank** (54 documents, 292 chunks). Retrieval metrics are near saturation, and real enterprise documents are longer and noisier.
- **Cross-lingual place names.** The Khobar showroom facts exist only in English ("Al Khobar"), and Arabic questions about them fail. "الخبر" is also a common Arabic word.
- **Saudi-dialect "how much".** "بكم" and "كم ياخذ" are sometimes answered with a product's dimensions instead of its price.
- **Strict abstention.** The 0.8445 gate, calibrated on validation and not changed afterwards, rejects some correctly retrieved mixed questions. The assessment's own example "أبغى أعرف الـ pricing حق الباقة المؤسسية" ranks the right chunk first, but scores 0.814 and is answered NOT_FOUND.
- **Unsupported claims with matching numbers.** The post-check verifies only numbers and codes. A showroom phone number presented as a WhatsApp number (`test_00017`) passes it.
- **Untranslated currency.** Some English answers built from Arabic sources keep "ريال سعودي" in Arabic.
- **Lenient LLM judge.** Its faithfulness and relevance figures are upper bounds. The human ratings (meaning 3.80, faithfulness 3.97 of 5) are lower and caught both known errors among the 30 items.
- **Small human evaluation.** It has one rater and 30 items, covers the after stack only, and its scores are coarse (mostly 4).
- **Rewriting.** Its latency is above the 200 ms target (see above). Its retrieval quality also trails the gold standalone question: second-turn Hit@1 is 0.899, against 0.972 for the gold question.
- **Suggested follow-ups.** They cost extra GPU time and compete with answer generation under load. They are not checked against the knowledge bank: in the demo run, one Arabic suggestion invented an entity ("ما رقم الهاتف الخاص بالمنشأة في عُمان؟"). Only their language and count are evaluated.
- **Non-deterministic decoding.** FP8 temperature-0 decoding varies slightly between identical runs.
- **Throughput ceiling.** Smart AI Search levels off at about 5.6 requests/s on one L4. The embedder and reranker share one lock, and the GPU is shared with vLLM.
- **Time to first token is measured on the server.** The API returns the answer after the post-checks, so clients see the full answer at once.
- **Deployment gaps.** The API has no authentication or rate limiting. The Docker Compose GPU clean start was validated on an RTX 4090, not an L4.
- **Licensing.** PyMuPDF is AGPL-3.0 (see Models and licenses).

## Possible improvements

- **Cross-lingual place names.** Add transliterated aliases ("الخبر" ↔ "Al Khobar") at indexing time, or as synthetic training pairs.
- **Dialect "how much".** Add dialect price paraphrases ("بكم", "كم ياخذ") as hard positives for price chunks, with dimension chunks as hard negatives.
- **Rewrite latency.** Move query rewriting to a smaller model, or cache rewrites for repeated follow-ups.
- **Streaming.** Stream tokens to the client and run the post-checks on the finished answer, retracting the answer if a check fails.
- **Abstention.** Calibrate the NOT_FOUND threshold per language bucket on a larger validation set, and add an entailment check for claims whose numbers do appear in the source.
- **Larger, real data.** Evaluate on real enterprise documents and longer multi-turn conversations.
- **Throughput.** Serve the embedder and reranker with ONNX or TensorRT, or on their own GPU stream, to lift the ~5 requests/s Smart AI Search ceiling on one L4.
