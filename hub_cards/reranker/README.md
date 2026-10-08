---
license: apache-2.0
base_model: BAAI/bge-reranker-v2-m3
language:
- ar
- en
library_name: sentence-transformers
pipeline_tag: text-ranking
tags:
- sentence-transformers
- cross-encoder
- reranker
- arabic
- cross-lingual
- rag
datasets:
- halarash/qimam-kb-rag-data
metrics:
- mrr
- ndcg
- recall
---

# Qimam KB bilingual reranker (bge-reranker-v2-m3, fine-tuned)

`halarash/qimam-bge-reranker-v2-m3-kb` is [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) fine-tuned as a cross-encoder for an Arabic/English enterprise knowledge bank. In the RAG system it reorders 30 fused BM25 + dense candidates, and its top score drives a calibrated "not found in knowledge bank" gate.

## Model details

| Field | Value |
|---|---|
| Base model | [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) |
| License | Apache-2.0, the same as the base model; commercial use is allowed |
| Architecture | XLM-RoBERTa large cross-encoder, 568M parameters, one output logit |
| Max sequence length | 512 tokens (query + passage) |
| Size on disk | 2.27 GB (float32 safetensors) |
| Serving precision | float16 on GPU, float32 on CPU |
| Runs on | GPU and CPU |
| Quantization | None |
| Output | A relevance score in [0, 1] (sigmoid of the logit) |

## Usage

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("halarash/qimam-bge-reranker-v2-m3-kb", max_length=512)
scores = model.predict([
    ("كم مدة ضمان المطابخ؟", "Warranty | Kitchens\nKitchens carry a 2-year warranty against manufacturing defects."),
    ("كم مدة ضمان المطابخ؟", "Warranty | Sofas\nSofa frames are covered by the warranty for 3 years."),
])
print(scores)
```

Passages are given as `title | section` followed by the chunk text, normalized as in the embedding model card.

## Training

| Item | Value |
|---|---|
| Method | Full fine-tuning with the sentence-transformers `CrossEncoderTrainer` |
| Loss | `BinaryCrossEntropyLoss` with `pos_weight = 4` to balance one positive against four negatives |
| Training rows | 1,438 (query, positive, 4 negatives) from 1,277 training questions; 7,190 scored pairs |
| Negatives per row | 3 hard negatives mined with the base embedding model (confusable facts first) and 1 from the BM25 top results |
| Batch size | 32 |
| Epochs | 1 (225 steps), no checkpoint selection |
| Learning rate | 1e-5, warmup ratio 0.1, weight decay 0.01, bf16 |
| Seed | 2026 |
| Hardware | 1 × NVIDIA A10G, 72.5 s of training |

**Leakage controls.** No negative states the query's own fact, so the translated twin of a positive is never a negative. No chunk tied to a validation or test fact appears in the training rows; the data builder fails if one does.

## Training data

[`halarash/qimam-kb-rag-data`](https://huggingface.co/datasets/halarash/qimam-kb-rag-data), files `training/reranker_train.jsonl` (training rows), `training/reranker_candidates.jsonl` (validation and test candidate pools) and `training/passages.json`.

## Evaluation

Reranking a 30-candidate pool; 153 validation and 238 test answerable questions. The reranker and candidate pool were selected on validation nDCG@10 only.

| System | Val nDCG@10 | Val MRR | Test nDCG@10 | Test MRR | Test Recall@1 |
|---|---|---|---|---|---|
| Fine-tuned dense, no reranker | 0.9521 | 0.9354 | 0.9763 | 0.9709 | 0.880 |
| Fine-tuned dense + base reranker | 0.9317 | 0.9125 | 0.9649 | 0.9592 | 0.868 |
| Fine-tuned dense + this reranker | 0.9740 | 0.9648 | 0.9880 | 0.9839 | 0.897 |
| Equal-weight hybrid + base reranker | 0.9291 | 0.9115 | 0.9663 | 0.9594 | 0.868 |
| **Equal-weight hybrid + this reranker (deployed)** | **0.9744** | **0.9654** | **0.9896** | **0.9860** | **0.901** |

- The base reranker lowers quality after the domain-tuned embedder; this reranker improves every candidate pool.
- The largest gains are on cross-lingual and dialect questions: test Arabic → English MRR rises from 0.872 to 0.935, and Saudi dialect MRR from 0.947 to 0.963.

## Limitations

- **Saturated scores.** BCE with `pos_weight` pushes most answerable top-1 scores above 0.99, so an abstention threshold sits on a steep part of the score curve. The deployed threshold (0.8445) was calibrated end to end on validation. Some correctly ranked mixed-language questions score below it, for example "أبغى أعرف الـ pricing حق الباقة المؤسسية" (0.814), and are answered NOT_FOUND.
- Arabic question → English passage remains the weakest direction (test MRR 0.935 on 28 questions).
- Saudi-dialect "how much" ("بكم", "كم ياخذ") is sometimes ranked toward a product's dimension passage instead of its price.
- One training run with no hyperparameter sweep, on synthetic single-company data.

## Reproduction

```bash
python scripts/prepare_data.py build-reranker-data
python scripts/train_reranker.py --hub-dataset halarash/qimam-kb-rag-data --push
python scripts/evaluate_reranker.py
```

Hyperparameters are in `configs/training_config.yaml`.
