# Before vs after fine-tuning

Both columns ran the same pipeline in the same NVIDIA L4 job: equal-weight RRF over BM25 and dense retrieval (30 candidates), cross-encoder reranking, a top-score NOT_FOUND gate, and Qwen3-4B-Instruct-2507-FP8 on vLLM with streaming. Only the two retrieval models change.

|  | Before | After |
|---|---|---|
| Embedder | Qwen/Qwen3-Embedding-0.6B | halarash/qimam-qwen3-embedding-0.6b-kb-v2 |
| Reranker | BAAI/bge-reranker-v2-m3 | halarash/qimam-bge-reranker-v2-m3-kb |
| NOT_FOUND gate threshold (validation-calibrated) | 0.0104 | 0.8445 |

Test split: 272 questions (238 answerable, 34 unanswerable). Retrieval rows use the deployed stack's reranked top 10 for each answerable question, grouped by question language; cross-lingual means the fact exists only in the other language. Latency rows are sequential, one request at a time, measured in process (no HTTP).

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
| Human meaning score (1–5) | – | – |
| Human fluency score, Arabic answers (1–5) | – | – |
| Human fluency score, English answers (1–5) | – | – |
| Time to first token, avg (ms) | 167 | 162 |
| Time to first token, p95 (ms) | 186 | 182 |
| Total latency, Smart AI, avg (ms) | 732 | 736 |
| Total latency, Smart AI, p95 (ms) | 1,217 | 1,229 |
| Quick Search p95 latency (ms, no fine-tuned component) | 25.0 | 25.0 |
| GPU memory used, whole stack (MiB) | 16,158 | 16,230 |
| GPU memory, embedder + reranker peak allocated (MiB) | 2,319 | 2,324 |
| CPU memory, host RAM used (MiB) | 13,361 | 13,743 |
| CPU memory, retrieval process RSS (MiB) | 2,460 | 2,782 |

Human scores are filled in from `results/human_eval/` once the ratings exist; a dash means not rated. The LLM-judge figures are bounds: on the known errors the Qwen3-8B judge is lenient (see `results/error_analysis.md`).
