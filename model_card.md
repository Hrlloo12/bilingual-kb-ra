# Model Card

This card covers every model used in the system. Every value marked *pending* has not been measured yet.

## 1. Fine-tuned embedding model: `halarash/qimam-qwen3-embedding-0.6b-kb-v2` (selected)

| Field | Value |
|---|---|
| Base model | [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) |
| License | Apache-2.0 (base and fine-tuned) |
| Commercial use | Allowed under Apache-2.0 |
| Parameters | 0.6B |
| Size on disk | 2.4 GB (float32 safetensors as pushed) |
| VRAM at serving | Measured together with the reranker on 1 × L4: 2.3 GB peak allocated (float16, both models plus activations). Per-model split pending (Day 5) |
| CPU | Yes. Used on CPU for indexing and evaluation in this project |
| GPU | Yes |
| Quantization | None |
| Embedding dimension | 1024, cosine similarity |
| Max sequence length | 512 tokens |
| Query prompt | `Instruct: Given a question in Arabic or English, retrieve knowledge bank passages that answer the question\nQuery:` (passages have no prompt) |
| Repository visibility | Private, access granted to reviewers on request |

### Training

| Item | Value |
|---|---|
| Method | Full fine-tuning of all transformer layers |
| Objective | `CachedMultipleNegativesRankingLoss` (InfoNCE with in-batch negatives plus two mined hard negatives per row) |
| Training rows | 1,438 (anchor, positive, negative_1, negative_2) built from 1,277 training queries |
| Batch size | 64 (gradient cache mini-batch 16) |
| Epochs | 3; checkpoint selected on validation nDCG@10 (best: epoch 1) |
| Learning rate | 1e-5, warmup ratio 0.1, weight decay 0.01, bf16 |
| Seed | 2026 |
| Hardware | 1 × NVIDIA A10G (Hugging Face Job), 453 s |
| Batch sampler | Group-aware: every row in a batch comes from a different fact group. Arabic and English chunks that state the same fact are therefore never in-batch negatives for each other |

**Hard-negative mining.** Negatives were mined with the base model over the indexed corpus:
- **Excluded:** chunks that state the query's own fact (so a cross-lingual twin of the positive is never a negative), and all chunks tied to validation or test facts.
- **Ranking:** chunks containing a fact marked confusable with the target (for example Premium vs Standard SLA, or bedroom vs kitchen warranty) come first, then by base-model similarity.

**Text normalization.** NFKC; removal of bidirectional control characters, tashkeel and tatweel; Arabic-Indic digits converted to ASCII. Applied identically to queries and passages. Passages are prefixed with their document title and section heading.

**Data augmentation.** Each fact has generated paraphrases in:
- Modern Standard Arabic (MSA) and long MSA;
- English and long English;
- Saudi dialect;
- code-switched Arabic.

Cross-lingual pairs come from facts that exist in only one language.

### Runs

| | Run 1 | Run 2 (selected) |
|---|---|---|
| Repository | `halarash/qimam-qwen3-embedding-0.6b-kb` | `halarash/qimam-qwen3-embedding-0.6b-kb-v2` |
| Batch sampler | First version: deferred rows formed many small tail batches | Corrected: one row per group per batch; epoch ends when fewer than 32 groups remain |
| Steps | 300 | 58 |
| Logged steps with zero loss | 11 of 60 | 0 of 11 |
| Hardware | NVIDIA L4 | NVIDIA A10G |
| Validation nDCG@10 / MRR@10 / Acc@1 | 0.9405 / 0.9203 / 0.8693 | **0.9486 / 0.9303 / 0.8758** |

Run 2 was selected on **validation metrics only**. The test set was not used to choose between the runs. Both runs' metrics are kept in `results/embedding_runs/`.

### Held-out test results (238 answerable queries, 292-chunk corpus)

| Dense retrieval | Base | Fine-tuned (run 2) |
|---|---|---|
| Recall@1 | 0.689 | 0.880 |
| Recall@5 | 0.954 | 0.996 |
| MRR | 0.847 | 0.971 |
| nDCG@10 | 0.871 | 0.976 |
| Cross-lingual MRR (93 queries) | 0.651 | 0.944 |

Full per-bucket results are in `results/before_finetuning_metrics.json` and `results/after_finetuning_metrics.json`.

### Known limitations
- **Synthetic, single-company training data.** Gains may be smaller on real enterprise documents with longer, noisier passages.
- **Small corpus and saturated metrics.** The corpus has 292 chunks, so Recall@5 is close to saturation and fine-grained differences between models are hard to measure.
- **Code-switched queries are under-represented:** 82 in train, 19 in test.
- **AR→EN cross-lingual is the weakest bucket:** test MRR 0.872 over 28 queries.
- **Only the test split was manually reviewed.** Training and validation queries passed automatic filters only.

## 2. Fine-tuned reranker: `halarash/qimam-bge-reranker-v2-m3-kb` (selected)

| Field | Value |
|---|---|
| Base model | [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) |
| License | Apache-2.0 (base and fine-tuned) |
| Commercial use | Allowed under Apache-2.0 |
| Parameters | 568M (XLM-RoBERTa large cross-encoder) |
| Size on disk | 2.27 GB (float32 safetensors as pushed) |
| Serving precision | float16 on GPU |
| Max sequence length | 512 tokens (query + passage) |
| Output | One logit per (query, passage) pair, passed through a sigmoid |
| Repository visibility | Private, access granted to reviewers on request |

### Training

| Item | Value |
|---|---|
| Method | Full fine-tuning, sentence-transformers `CrossEncoderTrainer` |
| Objective | `BinaryCrossEntropyLoss` with `pos_weight = 4` to balance the 1:4 positive/negative ratio |
| Training rows | 1,438 (query, positive, 4 negatives) from 1,277 training queries; 7,190 scored pairs |
| Negatives per row | 3 from the base-embedding hard-negative list (confusable facts first), 1 from BM25 top results |
| Batch size | 32 |
| Epochs | 1 (225 steps), no checkpoint selection |
| Learning rate | 1e-5, warmup ratio 0.1, weight decay 0.01, bf16 |
| Seed | 2026 |
| Hardware | 1 × NVIDIA A10G (Hugging Face Job), 72.5 s training |
| Logged loss | 0.90 at step 10, 0.08–0.25 from step 100 onward; no zero-loss steps |

**Leakage controls.** These are the same as for the embedding data:
- No negative states the query's own fact, so translated twins of the positive are never negatives.
- No chunk tied to a validation or test fact appears anywhere in the training rows. `prepare_data.py build-reranker-data` checks this and fails if one does.

**Selection.** The reranker and the candidate pool were chosen together on **validation nDCG@10 only**.

### Results (reranking a 30-candidate pool; 153 validation / 238 test answerable queries)

| System | Val nDCG@10 | Val MRR | Test nDCG@10 | Test MRR | Test Recall@1 |
|---|---|---|---|---|---|
| Fine-tuned dense, no reranker | 0.9521 | 0.9354 | 0.9763 | 0.9709 | 0.880 |
| Fine-tuned dense + base reranker | 0.9317 | 0.9125 | 0.9649 | 0.9592 | 0.868 |
| Fine-tuned dense + fine-tuned reranker | 0.9740 | 0.9648 | 0.9880 | 0.9839 | 0.897 |
| Equal hybrid + base reranker | 0.9291 | 0.9115 | 0.9663 | 0.9594 | 0.868 |
| **Equal hybrid + fine-tuned reranker (selected)** | **0.9744** | **0.9654** | **0.9896** | **0.9860** | **0.901** |

- **The base reranker makes ranking worse.** It lowers validation nDCG@10 from 0.952 to 0.932 and AR→EN validation MRR from 0.974 to 0.864. On this corpus, an off-the-shelf reranker after a domain-tuned embedder is harmful.
- **The fine-tuned reranker improves every system.** The largest gains are on cross-lingual and dialect queries: test AR→EN MRR rises from 0.872 to 0.935 and test dialect MRR from 0.947 to 0.963.
- **The pool choice is a near tie.** The equal-weight hybrid pool beat the dense-only pool by 0.0004 validation nDCG@10, about one query. Validation picked it under the pre-declared rule. Reranking makes BM25's ranking noise irrelevant, while BM25 still contributes lexical candidates.

Full results are in `results/reranker/` (`reranker_comparison_validation.json`, `reranker_comparison_test.json`, the raw scores and `training_metrics.json`).

### Known limitations
- **Scores are saturated.** BCE with `pos_weight` pushes most answerable top-1 scores above 0.99, so the abstention threshold sits on a steep part of the curve. It is calibrated end to end on validation (see Smart AI Search).
- **AR→EN remains the weakest bucket:** test MRR 0.935 over 28 queries.
- **One run, no hyperparameter sweep.**

## 3. Generator: `Qwen/Qwen3-4B-Instruct-2507-FP8` (not fine-tuned)

| Field | Value |
|---|---|
| Model | [Qwen/Qwen3-4B-Instruct-2507-FP8](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507-FP8), official FP8 release (fine-grained block-wise FP8 weights) |
| License | Apache-2.0 |
| Commercial use | Allowed under Apache-2.0 |
| Parameters | 4.0B |
| Mode | Instruct (non-thinking); no `<think>` output |
| Serving | vLLM 0.10.1.1 (torch 2.7.1+cu126, transformers 4.55.4), OpenAI-compatible `/v1/chat/completions` |
| Decoding | temperature 0, max 384 new tokens, seed 0 |
| Context | `--max-model-len 8192`; prompts average 260 tokens (system prompt + up to 4 passages + question) |
| VRAM on 1 × L4 (measured) | Weights 4.23 GiB; KV cache 6.47 GiB (47,120 tokens); total vLLM reservation 13.3 GB at `--gpu-memory-utilization 0.55` |
| Fine-tuning | None, by design. Grounding is enforced by retrieval, the prompt, deterministic citations and post-checks |

**Prompt contract.** The system prompt requires the model to:
- use only the numbered passages;
- end every sentence with a passage number such as `[1]`;
- copy numbers, prices, dates and codes exactly;
- answer in the user's language (mixed queries are answered in Arabic, keeping English terms);
- reply with exactly `NOT_FOUND` when the passages do not contain the answer.

The model only chooses among the passage numbers it was given. Every citation field (document, chunk, title, section, page, source) comes from the retrieved chunk's metadata, never from model output.

**Measured on 1 × NVIDIA L4** (final Day 3 run `day3_l4_midpoint_20261007-1544`; 272 test queries, sequential, one request at a time):

| | avg | p50 | p95 | max |
|---|---|---|---|---|
| Generation latency (ms), 238 generated answers | 707 | 680 | 1,109 | 1,472 |
| Completion length (tokens) | 28.6 | | | |

**Determinism.** Temperature 0 is not perfectly reproducible with FP8 kernels. Two runs with identical retrieval and generation settings produced differently worded answers for 36 of 272 test queries. One unanswerable validation query flipped from NOT_FOUND to an unsupported answer.

## 4. Query generation model (dataset construction only)

| Field | Value |
|---|---|
| Model | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), Apache-2.0 |
| Use | Offline generation of query paraphrases; not part of the serving system |
| Serving | vLLM 0.10.1.1 inside a Hugging Face Job (1 × L4), JSON-constrained decoding, temperature 0.8, seed 13, thinking disabled |
