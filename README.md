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
| End-to-end Smart AI Search on the L4 | `TASK=smart_search scripts/run_l4_job.sh` (HF Job `l4x1`) | `results/smart_search/` |
| Interactive AI Search evaluation on the L4 | `TASK=interactive scripts/run_l4_job.sh` (HF Job `l4x1`) | `results/interactive/` |

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

Pipeline:

1. Equal-weight RRF over BM25 top 50 and fine-tuned dense top 50 gives 30 candidates.
2. The fine-tuned reranker reorders them, and a top-score gate can abstain at this point.
3. Up to 4 passages with reranker score ≥ 0.5 go to Qwen3-4B-Instruct-2507-FP8 (vLLM 0.10.1.1).
4. Citations are mapped to chunk metadata, then number/code/language post-checks run.

All runs used HF Jobs `l4x1`, with queries sent sequentially, one at a time.

| Run | Purpose | Folder |
|---|---|---|
| `day3_l4_20261007-1513` | Ungated validation run used for calibration, then a first gated run at threshold 0.8887 | `results/smart_search/day3_l4_20261007-1513/` |
| `day3_l4_midpoint_20261007-1544` | **Final Day 3 measurement**, gated at the corrected midpoint threshold 0.844483 | `results/smart_search/day3_l4_midpoint_20261007-1544/` |

**Abstention calibration (validation only).**
- Validation was first run with no gate. The gate threshold was then chosen to maximise the balanced accuracy of the final answered/NOT_FOUND decision.
- Candidate thresholds are midpoints between observed top scores. This avoids a float-equality edge case in the first calibration, which picked 0.8887, exactly one query's score.
- Without the gate, the generator alone returned NOT_FOUND for 88.5% of unanswerable validation questions and answered 97.4% of answerable ones.
- On that run, the gate predicted 100% NOT_FOUND on unanswerable questions while answering 95.4% of answerable ones.
- When measured again, validation gave 94.8% answered and 25/26 NOT_FOUND. The miss is a generator non-determinism case (see below).

| Test split (238 answerable, 34 unanswerable), final run | Value |
|---|---|
| Answerable questions answered | 0.979 (233 / 238) |
| Unanswerable questions → NOT_FOUND | 0.971 (33 / 34) |
| Relevant chunk in the generator's context | 0.971 |
| Answers whose citations include a relevant chunk | 0.987 |
| Citation precision (cited chunks that are relevant) | 0.976 |
| Answers with no valid citation marker (fallback used) | 0.000 |
| Answer in the expected language | 0.991 |
| Reference numbers all present in the answer (product codes excluded) | 0.897 (n = 175) |
| Answer numbers all present in the reference | 0.983 (n = 171) |
| Answers rejected by the number/code post-check | 0 |

**Known failures found in these runs** (to be expanded in the Day 5 error analysis):
- **Unanswerable questions that were answered.**
  - `test_00017` asked for the customer-service WhatsApp number. It was answered with the Abha showroom phone number, labelled as WhatsApp.
  - `validation_00015` asked about the year-end bonus. It was answered with annual-leave rules. The same query returned NOT_FOUND in the earlier identical run.
  - The number post-check cannot catch either case, because the numbers do appear in the cited passage.
- **Dialect price questions answered with dimensions.** Three Saudi-dialect price questions ("بكم", "كم ياخذ") got the product's dimensions instead of its price.
- **Khobar showroom never found.** All five false NOT_FOUNDs on test concern this fact (English-only, asked in Arabic). Retrieval missed the chunk and the generator refused rather than guessed.
- **Non-determinism.** Temperature-0 decoding with FP8 kernels is not fully reproducible: 36 of 272 test answers were worded differently between the two runs, and one validation decision flipped.

Latency in ms (all 272 test queries, final run; queries stopped by the gate skip generation):

| Stage | avg | p50 | p95 | max |
|---|---|---|---|---|
| BM25 | 5.9 | 6.0 | 7.4 | 14.0 |
| Query embedding | 35.4 | 34.9 | 38.6 | 47.2 |
| Dense search | 4.9 | 4.8 | 5.5 | 6.3 |
| Rerank (30 pairs) | 45.1 | 44.3 | 58.9 | 68.4 |
| Generation (238 queries) | 707 | 680 | 1,109 | 1,472 |
| **Total** | **710** | **740** | **1,173** | **1,575** |

**GPU memory.** Peak use was 16.2 GB of 23.0 GB:
- vLLM reservation: 13.3 GB (`--gpu-memory-utilization 0.55`).
- Embedder and reranker: 2.3 GB peak allocated.

These are single-user latencies. Concurrency, corpus scaling and the full generation-quality evaluation are Day 5 work.

### Interactive AI Search and deployment (Day 4)

**API** (one surface for all three modes, served on port 8080):

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/search` | `{"mode": "quick_search" \| "smart_search" \| "interactive", "query": "...", "session_id": "...", "top_k": n}` |
| `GET` | `/v1/sessions/{id}` | Session turns and remaining TTL |
| `DELETE` | `/v1/sessions/{id}` | End a conversation |
| `GET` | `/health` | Readiness of OpenSearch, Qdrant, Valkey and the generator (503 until all are ready) |
| `GET` | `/livez` | Liveness |
| `GET` | `/` | Minimal UI (three modes, RTL-aware, citations, conversation view) |

**How interactive search works.**
- **Memory:** each session keeps its last 6 turns in Valkey with a 30-minute TTL.
- **Skip gate:** a rule-based check, with no model call, decides whether the new question depends on the previous turn. It looks for follow-up openers, back-references, attached Arabic pronouns and very short questions, and treats long questions that name their own product or place as standalone.
- **Rewrite:** a dependent question is rewritten by Qwen3-4B into a standalone question in the user's own language, using only the previous turn. If the rewrite fails, the previous question and the follow-up are concatenated instead.
- **Answer:** the standalone question then goes through the unchanged Smart AI Search pipeline, and the answer follows the language of the user's follow-up.
- **Tuning:** the gate rules were adjusted on validation conversations only.

**Evaluation set.** `scripts/build_conversations.py` builds 573 two-turn conversations (227 validation, 346 test). The second turn always targets a held-out fact.

| Group | Contents |
|---|---|
| Conversation types | Attribute follow-ups ("And how much does it cost?"), bare follow-ups ("Price?"), entity switches ("What about the Rimal bed?"), and standalone second turns taken from the dataset queries |
| Language patterns | EN→EN, AR→AR, EN→AR and AR→mixed |

**Results on 1 × NVIDIA L4 before the rewrite fix** (run `day4_l4_interactive_20261007-1939`, HF Job `l4x1`, sequential requests through the API). Test split:

| Second-turn system | Hit@1 | Hit@5 | MRR@10 | Answered | Cites a relevant chunk |
|---|---|---|---|---|---|
| No rewrite (raw follow-up to Smart Search) | 0.341 | 0.541 | 0.448 | 0.448 | 0.292 |
| **Interactive (gate + rewrite + memory)** | **0.887** | **0.965** | **0.922** | **0.879** | **0.841** |
| Oracle (gold standalone question) | 0.972 | 1.000 | 0.981 | 0.934 | 0.934 |

| Gate and rewrite (test) | Value |
|---|---|
| Follow-ups sent to rewrite (recall) | 1.000 (286 / 286) |
| Standalone questions rewritten unnecessarily | 0.017 (1 / 60) |
| Rewrite fallbacks | 0 |
| Rewrites kept in the user's language | 0.990 |
| Rewrite latency (ms) avg / p50 / p95 / max | 416 / 354 / 688 / 811 |
| Memory read + write on the skip path (ms, avg) | 0.5 |
| Interactive turn total (ms) avg / p50 / p95 | 1,084 / 1,003 / 1,843 |

**Rewrite prompt fix (selected on validation, test run once).**

The first prompt (`v1`) had a single Arabic entity-switch example, and it asked about price. English-to-Arabic topic switches therefore often carried "price" into the rewrite even when the user had asked about dimensions.

The fix:
- Two candidates were written, each with six balanced examples (price, dimensions, warranty, availability, location/phone, installation time) using names absent from the knowledge bank. `v2` adds a "keep the same attribute" rule; `v3` uses the examples alone.
- All three prompts were scored on the 227 validation conversations only (`results/interactive/day4_rewrite_candidates_20261007-2232/`).
- `v2` was selected by the pre-declared rule: highest validation MRR@10 (v1 0.9215, **v2 0.9441**, v3 0.9439). `v2` and `v3` are practically tied; the balanced examples drive the gain.
- The prompt was then frozen (`interactive.rewrite_prompt: v2`) and the test set was run once.

| Test (346 conversations, NVIDIA L4) | Before (`v1`) | After (`v2`) |
|---|---|---|
| Hit@1 | 0.887 | **0.913** |
| MRR@10 | 0.922 | **0.940** |
| Answered | 0.879 | **0.902** |
| Cites a relevant chunk | 0.841 | **0.882** |
| EN→AR Hit@1 / MRR@10 | 0.747 / 0.817 | **0.853 / 0.894** |
| AR→mixed Hit@1 / MRR@10 | 0.864 / 0.904 | **0.898 / 0.938** |
| AR→AR Hit@1 / MRR@10 | 0.885 / 0.928 | 0.875 / 0.914 |
| EN→EN Hit@1 | 1.000 | 1.000 |
| Rewrite latency p50 / p95 (ms) | 354 / 688 | 338 / 675 |
| Rewrites kept in the user's language | 0.990 | 0.993 |
| Answers in the expected language | 0.993 | 0.994 |

- The prompt-independent baselines did not change (no rewrite 0.341, oracle 0.972 Hit@1), so the two runs are comparable.
- AR→AR dropped by about one query, and bare follow-ups by two; both are reported rather than tuned away.
- Before: `day4_l4_interactive_20261007-1939`. After: `day4_l4_interactive_v2_20261007-2302`.

**Clean-start deployment (validated on an RTX 4090 VM).** The full Docker Compose clean start was validated on a Vast.ai VM with an RTX 4090 24 GB, Ubuntu 22.04, Docker 28.1 and Compose 2.35, using `scripts/clean_start_test.sh`.

The official performance numbers are measured on the required NVIDIA L4 24 GB through Hugging Face Jobs. HF Jobs cannot run Docker Compose, so the same services and versions (OpenSearch 2.19.1, Qdrant 1.14.1, Valkey 8.1, vLLM 0.10.1.1, the same API code and configuration) run there as processes (`scripts/run_l4_job.sh`). It removes all containers, volumes and images, rebuilds with no cache, starts the stack and runs the smoke test.

| Clean-start stage | Result |
|---|---|
| Image build (no cache) | 305 s |
| Start to all services healthy | 1,053 s (image pulls, model downloads, CUDA graph capture) |
| Smoke checks | 16 / 17 |
| GPU memory with the full stack | 17.3 GB of 24.6 GB |
| Host RAM, all containers | ≈ 8.5 GB |

Two attempts failed before this one, and both causes are fixed: the indexer lacked write permission in the image, and OpenSearch was not yet writable at first index creation. A third issue (a results bind-mount permission) prevented the first `smoke.json` from being written; the smoke output is kept in the run log. No Vast.ai L4 offered VM mode with Docker, so the official L4 measurements were run as an HF Job with the same stack as processes.

**L4 resource use** (interactive run):
- GPU peak 16.2 GB of 23.0 GB:
  - vLLM: 13.3 GB;
  - API models: about 2.4 GB.
- GPU utilisation: 90% average while busy.
- Host CPU: 16% average, 60% peak.
- Host RAM peak: 10.9 GB, of which:
  - vLLM: 4.2 GB;
  - API: 3.3 GB;
  - OpenSearch: 1.4 GB;
  - Qdrant: 0.1 GB;
  - Valkey: 0.02 GB.

**Smoke failure (unchanged threshold).** The mixed query "أبغى أعرف الـ pricing حق الباقة المؤسسية" ranks the correct chunk first, but its reranker score is 0.814 on the L4, below the validation-calibrated threshold 0.8445, so it returns NOT_FOUND. The threshold was not tuned on this query.
