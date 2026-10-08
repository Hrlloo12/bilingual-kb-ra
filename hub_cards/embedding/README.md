---
license: apache-2.0
base_model: Qwen/Qwen3-Embedding-0.6B
language:
- ar
- en
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- feature-extraction
- retrieval
- arabic
- cross-lingual
- rag
datasets:
- halarash/qimam-kb-rag-data
metrics:
- recall
- mrr
- ndcg
---

# Qimam KB bilingual embedder (Qwen3-Embedding-0.6B, fine-tuned)

`halarash/qimam-qwen3-embedding-0.6b-kb-v2` is [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) fine-tuned for retrieval over a bilingual Arabic/English enterprise knowledge bank. It is the dense retriever of a self-hosted RAG system with three search modes (Quick Search, Smart AI Search, Interactive AI Search).

It was trained for Modern Standard Arabic, English, Saudi dialect and code-switched Arabic/English questions, including cross-lingual retrieval (Arabic question → English passage and the reverse).

## Model details

| Field | Value |
|---|---|
| Base model | [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) |
| License | Apache-2.0, the same as the base model; commercial use is allowed |
| Parameters | 0.6B |
| Embedding dimension | 1024, cosine similarity (normalized embeddings) |
| Max sequence length | 512 tokens |
| Size on disk | 2.4 GB (float32 safetensors) |
| Serving precision | float16 on GPU, float32 on CPU |
| Runs on | GPU and CPU |
| Quantization | None |

## Usage

Queries use an instruction prompt; passages do not.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("halarash/qimam-qwen3-embedding-0.6b-kb-v2")
prompt = "Instruct: Given a question in Arabic or English, retrieve knowledge bank passages that answer the question\nQuery:"
queries = model.encode(["كم مدة ضمان المطابخ؟"], prompt=prompt, normalize_embeddings=True)
passages = model.encode(["Warranty | Kitchens\nKitchens carry a 2-year warranty against manufacturing defects."], normalize_embeddings=True)
print(queries @ passages.T)
```

In the system, passages are prefixed with `title | section` and normalized the same way as queries: NFKC, removal of bidirectional control characters, tashkeel and tatweel, and Arabic-Indic digits converted to ASCII.

## Training

| Item | Value |
|---|---|
| Method | Full fine-tuning of all transformer layers |
| Loss | `CachedMultipleNegativesRankingLoss` (InfoNCE with in-batch negatives plus two mined hard negatives per row) |
| Training rows | 1,438 (anchor, positive, negative_1, negative_2) from 1,277 training questions |
| Batch size | 64 (gradient-cache mini-batch 16) |
| Epochs | 3, checkpoint selected on validation nDCG@10 (best: epoch 1) |
| Learning rate | 1e-5, warmup ratio 0.1, weight decay 0.01, bf16 |
| Seed | 2026 |
| Hardware | 1 × NVIDIA A10G, 453 s |
| Batch sampler | Group-aware: each batch holds at most one row per fact group, so the Arabic and English chunks that state the same fact are never in-batch negatives for each other |

**Hard negatives** were mined with the base model over the indexed corpus. Chunks that state the query's own fact (including its translation) and every chunk tied to a validation or test fact are excluded. Chunks holding facts marked as confusable with the target (for example Premium vs Standard SLA) are ranked first.

**Data augmentation.** Each fact has generated paraphrases in MSA, long MSA, English, long English, Saudi dialect and code-switched Arabic. Cross-lingual pairs come from facts that exist in only one language.

## Training data

[`halarash/qimam-kb-rag-data`](https://huggingface.co/datasets/halarash/qimam-kb-rag-data): a synthetic knowledge bank for a fictional home-furnishing company (54 documents, 292 chunks) and generated questions split by fact group into train (1,277), validation (179) and test (272) with no fact shared across splits. See the dataset card for generation and review details.

## Evaluation

Held-out test split, 238 answerable questions, 292-chunk corpus, dense retrieval only.

| Metric | Base model | This model |
|---|---|---|
| Recall@1 | 0.689 | 0.880 |
| Recall@5 | 0.954 | 0.996 |
| Recall@10 | 0.983 | 0.996 |
| Recall@20 | 0.996 | 1.000 |
| MRR | 0.847 | 0.971 |
| nDCG@10 | 0.871 | 0.976 |
| Cross-lingual MRR (93 questions) | 0.651 | 0.944 |
| Arabic question → English passage MRR (28) | 0.500 | 0.872 |
| English question → Arabic passage MRR (46) | 0.779 | 1.000 |
| Saudi dialect MRR (41) | 0.842 | 0.947 |

Run selection (run 1 vs run 2 of this repository) used validation metrics only.

## Limitations

- Trained and evaluated on one synthetic company's documents. Gains on real, longer and noisier enterprise documents will likely be smaller.
- The test corpus is small (292 chunks), so Recall@5 is close to saturation.
- Arabic question → English passage is the weakest direction (MRR 0.872 on 28 questions). Place names that are also common Arabic words (for example "الخبر") are a known failure.
- Code-switched questions are under-represented (82 in train, 19 in test).
- Only the test split was reviewed by hand; training and validation questions passed automatic filters.

## Reproduction

The training script, configuration and data preparation are in the project repository:

```bash
python scripts/prepare_data.py split-facts
python scripts/generate_queries.py --repo-id halarash/qimam-kb-rag-data
python scripts/prepare_data.py build-datasets
python scripts/prepare_data.py mine-negatives
python scripts/train_embeddings.py --hub-dataset halarash/qimam-kb-rag-data --push
```

Hyperparameters are in `configs/training_config.yaml`.
