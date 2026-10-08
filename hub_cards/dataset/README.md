---
license: cc-by-4.0
language:
- ar
- en
task_categories:
- text-retrieval
- question-answering
tags:
- rag
- arabic
- cross-lingual
- synthetic
- enterprise-search
pretty_name: Qimam Home bilingual knowledge bank RAG dataset
size_categories:
- 1K<n<10K
---

# Qimam Home bilingual knowledge bank (RAG dataset)

A synthetic Arabic/English enterprise knowledge bank with retrieval, answer and conversation evaluation sets. It was built to fine-tune and evaluate a self-hosted RAG system:
- the embedder [`halarash/qimam-qwen3-embedding-0.6b-kb-v2`](https://huggingface.co/halarash/qimam-qwen3-embedding-0.6b-kb-v2);
- the reranker [`halarash/qimam-bge-reranker-v2-m3-kb`](https://huggingface.co/halarash/qimam-bge-reranker-v2-m3-kb).

**Everything is fictional.** Qimam Home (قمم للمنزل) is an invented Saudi home-furnishing company. All names, products, prices, phone numbers, addresses and policies are made up.

## Contents

| Path | Contents |
|---|---|
| `facts/*.yaml` | 299 ground-truth facts in 14 domains (hand-written facts plus entity tables expanded into facts), each with an Arabic and English statement, numbers, and the languages it appears in |
| `corpus/raw/` | 54 documents rendered from the facts: 20 Arabic, 18 English, 16 mixed Arabic/English; 15 PDF, 14 DOCX, 16 HTML, 9 TXT; 6 use Eastern Arabic digits (٠–٩) |
| `corpus/raw/manifest.jsonl` | For every document section, the fact ids it states (the relevance labels) |
| `corpus/processed/chunks.jsonl` | The 292 chunks exactly as indexed (doc id, chunk id, title, section, page, language, source, text) |
| `train.jsonl`, `validation.jsonl`, `test.jsonl` | Questions with `query`, `language`, `bucket`, `relevant_fact_ids` and `reference_answer` |
| `unanswerable_queries.jsonl` | Hand-written questions the knowledge bank cannot answer (validation and test only) |
| `hard_negatives.jsonl` | Mined negatives for each training question, with base-model scores and a `confusable` flag |
| `training/embedding_train.jsonl` | Embedder rows: anchor, positive, negative_1, negative_2 |
| `training/reranker_train.jsonl`, `training/reranker_candidates.jsonl`, `training/passages.json` | Reranker rows and the validation/test candidate pools |
| `training/ir_validation.json` | Validation queries, corpus and relevance used for checkpoint selection |
| `interactive/conversations_validation.jsonl`, `interactive/conversations_test.jsonl` | 227 and 346 two-turn conversations for Interactive AI Search |
| `generation/fact_prompts.jsonl`, `generation/raw_queries.jsonl` | Query-generation input and the raw model output before filtering |
| `review/test_review.yaml` | The manual review of the test split and its 17 corrections |
| `templates/` | The Jinja templates that render the documents from the facts |
| `splits/fact_splits.json`, `dataset_report.json`, `smoke_queries.jsonl` | Fact-group split assignment, dataset statistics, and the deployment smoke-test questions |

The `code/` and `runs/` folders hold code snapshots and outputs of the benchmark jobs that used this dataset.

## Splits

Facts that share a document section form a group. Whole groups are assigned to one split (seed 2026), so no fact and no positive chunk appears in more than one split.

| Split | Questions | Facts | AR | AR dialect | EN | Mixed | AR → EN | EN → AR | Unanswerable |
|---|---|---|---|---|---|---|---|---|---|
| train | 1,277 | 228 | 326 | 221 | 212 | 82 | 209 | 227 | 0 |
| validation | 179 | 27 | 46 | 27 | 19 | 8 | 19 | 34 | 26 |
| test | 272 | 41 | 68 | 41 | 36 | 19 | 28 | 46 | 34 |

Buckets:
- `ar_ar`, `en_en`: same-language questions.
- `dialect`: Saudi dialect.
- `mixed`: code-switched Arabic/English.
- `ar_en`, `en_ar`: cross-lingual. For example, `ar_en` is an Arabic question whose fact exists only in English documents.

Interactive conversations follow the same split. Their second turn always targets a held-out fact, and each record carries a gold standalone rewrite.

## How it was made

1. **Facts.** Written by hand, with entity tables (products, showrooms, grades) expanded into facts. Facts that are easy to confuse (Premium vs Standard SLA, bedroom vs kitchen warranty) are linked as `confusable_with`.
2. **Documents.** Rendered from Jinja templates into PDF (WeasyPrint), DOCX, HTML and TXT, with Arabic, English and mixed documents and some Eastern Arabic digits.
3. **Questions.** For each fact, [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) (Apache-2.0) on vLLM wrote six questions: MSA, long MSA, English, long English, Saudi dialect and code-switched Arabic. Decoding was JSON-constrained with temperature 0.8 and seed 13.
4. **Automatic filtering.** 1,668 questions were accepted. Rejected: 56 that contained the answer, 36 in the wrong script and 34 duplicates. 259 were relabelled to the bucket their text actually matched.
5. **Manual review.** Every answerable test question was read against its fact. 17 were corrected for broken wording, a false premise or an embedded answer (`review/test_review.yaml`). Training and validation questions passed the automatic filters only.
6. **Unanswerable questions** were written by hand for validation and test.
7. **Hard negatives** were mined with the base embedder over train-split chunks only. A chunk that states the query's own fact is never used as a negative.

## Intended use

Fine-tuning and evaluating bilingual Arabic/English retrievers, rerankers and grounded generators, including cross-lingual retrieval and "not found" abstention.

## Limitations

- **Synthetic and small** (54 documents, 292 chunks): retrieval metrics saturate quickly, and real enterprise documents are longer and noisier.
- **Generated questions.** They inherit the style of one model. Code-switched questions are relatively few (82 in train).
- **Review coverage.** Only the test split was reviewed by hand.
- **Two-turn conversations only**, built from templates plus dataset questions.

## License

The dataset is released under CC-BY-4.0. Questions were generated with Qwen3-8B, which is Apache-2.0 licensed. Everything else was written for this project.

## Reproduction

The preparation scripts are in the project repository:

```bash
python scripts/prepare_data.py validate-facts
python scripts/prepare_data.py render-kb
python scripts/prepare_data.py split-facts
python scripts/generate_queries.py --repo-id halarash/qimam-kb-rag-data
python scripts/prepare_data.py build-datasets
python scripts/prepare_data.py mine-negatives
python scripts/prepare_data.py build-reranker-data
python scripts/build_conversations.py
```
