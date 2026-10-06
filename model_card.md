# Model Card

This card covers every model used in the system. The reranker and generator sections are filled in once those components are integrated and measured. Every value marked *pending* has not been measured yet.

## 1. Fine-tuned embedding model: `halarash/qimam-qwen3-embedding-0.6b-kb-v2` (selected)

| Field | Value |
|---|---|
| Base model | [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) |
| License | Apache-2.0 (base and fine-tuned) |
| Commercial use | Allowed under Apache-2.0 |
| Parameters | 0.6B |
| Size on disk | 2.4 GB (float32 safetensors as pushed) |
| VRAM at serving | pending, measured on the L4 benchmark |
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

## 2. Reranker: `BAAI/bge-reranker-v2-m3`
pending (Day 3)

## 3. Generator: `Qwen/Qwen3-4B-Instruct-2507-FP8`
pending (Day 3)

## 4. Query generation model (dataset construction only)

| Field | Value |
|---|---|
| Model | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), Apache-2.0 |
| Use | Offline generation of query paraphrases; not part of the serving system |
| Serving | vLLM 0.10.1.1 inside a Hugging Face Job (1 × L4), JSON-constrained decoding, temperature 0.8, seed 13, thinking disabled |
