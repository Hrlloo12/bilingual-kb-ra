# Latency report

**Hardware (GPU run):** CPU Intel(R) Xeon(R) Platinum 8268 CPU @ 2.90GHz, 24 usable vCPUs, 257,863 MiB RAM; GPU NVIDIA L4 (23,034 MiB, driver 595.71.05); Linux-6.8.0-139-generic-x86_64-with-glibc2.35.

Requests were sent one at a time through the HTTP API (`scripts/benchmark_latency.py api`). Client time includes HTTP, JSON serialization and the full pipeline; server stages come from each response's `latency_ms`. Time to first token is measured server-side from the moment the request is received until vLLM streams the first answer token; the API returns the answer only after the number/citation post-checks, so it does not stream tokens to the client.

## End to end, per mode (ms)

| Mode | Requests | avg | p50 | p95 | max | First result p50 | First result p95 | First token p50 | First token p95 |
|---|---|---|---|---|---|---|---|---|---|
| Quick Search | 272 | 17.7 | 17.1 | 25.0 | 47.2 | 13.1 | 20.2 | – | – |
| Smart AI Search | 100 | 769 | 824 | 1,260 | 1,437 | 78.6 | 92.5 | 176 | 192 |
| Interactive AI Search | 100 | 1,264 | 1,232 | 1,815 | 2,266 | 82.4 | 706 | 222 | 800 |

Targets: Quick Search p95 < 100 ms; Smart AI Search first token < 1,500 ms and total < 4,000 ms at p95; Interactive rewrite < 200 ms.

## Quick Search: stages (server, ms)

| Stage | avg | p50 | p95 | max |
|---|---|---|---|---|
| preprocessing | 0.1 | 0.1 | 0.1 | 0.7 |
| search | 13.6 | 13.0 | 20.1 | 42.6 |
| total | 13.8 | 13.2 | 20.3 | 42.9 |

## Smart AI Search: stages (server, ms)

| Stage | avg | p50 | p95 | max |
|---|---|---|---|---|
| preprocessing | 0.0 | 0.0 | 0.1 | 0.1 |
| bm25 | 18.4 | 17.8 | 26.1 | 38.2 |
| embedding | 49.8 | 49.3 | 56.3 | 63.0 |
| dense | 11.1 | 10.8 | 13.7 | 16.9 |
| fusion | 0.2 | 0.2 | 0.4 | 0.6 |
| retrieval | 79.4 | 78.6 | 92.5 | 107 |
| rerank | 52.2 | 51.7 | 62.9 | 71.7 |
| generation_first_token | 43.2 | 42.8 | 48.5 | 50.3 |
| time_to_first_token | 176 | 176 | 192 | 211 |
| generation_total | 725 | 729 | 1,142 | 1,303 |
| post_checks | 0.6 | 0.6 | 0.8 | 1.2 |
| total | 764 | 819 | 1,255 | 1,433 |

## Interactive AI Search: stages (server, ms)

| Stage | avg | p50 | p95 | max |
|---|---|---|---|---|
| memory_read | 0.6 | 0.5 | 0.8 | 0.9 |
| query_rewrite | 174 | 0.0 | 629 | 686 |
| preprocessing | 0.0 | 0.0 | 0.0 | 0.0 |
| bm25 | 16.2 | 15.9 | 22.3 | 34.5 |
| embedding | 50.5 | 49.0 | 57.1 | 118 |
| dense | 10.9 | 10.3 | 13.2 | 27.5 |
| fusion | 0.2 | 0.1 | 0.4 | 0.4 |
| retrieval | 77.8 | 76.6 | 87.1 | 161 |
| rerank | 47.1 | 47.0 | 55.5 | 65.9 |
| generation_first_token | 82.4 | 82.8 | 99.3 | 102 |
| time_to_first_token | 368 | 222 | 800 | 897 |
| generation_total | 769 | 773 | 1,116 | 1,427 |
| post_checks | 0.6 | 0.6 | 0.7 | 0.8 |
| followups | 930 | 880 | 1,431 | 1,572 |
| followups_wait | 242 | 187 | 791 | 1,153 |
| smart_search | 841 | 836 | 1,219 | 1,584 |
| memory_write | 0.8 | 0.8 | 1.1 | 1.8 |
| total | 1,259 | 1,227 | 1,810 | 2,260 |

Follow-up turns only (turn 2, where the rewrite runs):

| Stage | avg | p50 | p95 | max |
|---|---|---|---|---|
| memory_read | 0.6 | 0.6 | 0.8 | 0.9 |
| query_rewrite | 349 | 337 | 662 | 686 |
| preprocessing | 0.0 | 0.0 | 0.0 | 0.0 |
| bm25 | 16.1 | 15.4 | 23.9 | 32.1 |
| embedding | 50.0 | 49.1 | 56.2 | 63.5 |
| dense | 10.5 | 10.3 | 12.5 | 13.4 |
| fusion | 0.2 | 0.1 | 0.3 | 0.4 |
| retrieval | 76.7 | 76.3 | 84.8 | 91.2 |
| rerank | 47.3 | 46.9 | 58.3 | 65.9 |
| generation_first_token | 81.4 | 82.1 | 98.6 | 102 |
| time_to_first_token | 545 | 534 | 858 | 897 |
| generation_total | 717 | 663 | 1,059 | 1,079 |
| post_checks | 0.6 | 0.6 | 0.7 | 0.7 |
| followups | 863 | 792 | 1,394 | 1,484 |
| followups_wait | 260 | 148 | 766 | 1,153 |
| smart_search | 756 | 763 | 1,179 | 1,192 |
| memory_write | 0.8 | 0.8 | 1.0 | 1.8 |
| total | 1,366 | 1,295 | 2,111 | 2,260 |

## Smart AI Search in process, all 272 test questions (ms)

No HTTP; queries stopped by the NOT_FOUND gate skip generation.

| Stage | avg | p50 | p95 | max |
|---|---|---|---|---|
| preprocessing | 0.0 | 0.0 | 0.1 | 0.1 |
| bm25 | 11.8 | 11.4 | 17.8 | 31.3 |
| embedding | 47.8 | 47.9 | 55.6 | 63.7 |
| dense | 10.0 | 9.6 | 13.6 | 15.8 |
| fusion | 0.1 | 0.1 | 0.3 | 0.4 |
| retrieval | 69.7 | 69.8 | 82.3 | 100 |
| rerank | 48.6 | 47.9 | 62.4 | 71.3 |
| generation_first_token | 41.3 | 40.8 | 44.6 | 53.2 |
| time_to_first_token | 162 | 160 | 182 | 198 |
| generation_total | 705 | 681 | 1,126 | 1,425 |
| post_checks | 0.6 | 0.6 | 1.0 | 1.3 |
| total | 736 | 774 | 1,229 | 1,570 |

## Maximum input length tested

The API accepts up to 1,000 characters per question. Each input was sent 3 modes × 5 times.

| Input | Characters | Mode | avg ms | max ms | Max prompt tokens |
|---|---|---|---|---|---|
| longest arabic test query | 152 | Quick Search | 12.2 | 13.3 | – |
| longest arabic test query | 152 | Smart AI Search | 918 | 923 | 353 |
| longest arabic test query | 152 | Interactive AI Search | 1,730 | 1,737 | 353 |
| longest english test query | 153 | Quick Search | 14.0 | 16.4 | – |
| longest english test query | 153 | Smart AI Search | 524 | 529 | 326 |
| longest english test query | 153 | Interactive AI Search | 1,164 | 1,180 | 326 |
| max length input | 981 | Quick Search | 18.1 | 21.6 | – |
| max length input | 981 | Smart AI Search | 275 | 283 | – |
| max length input | 981 | Interactive AI Search | 1,120 | 1,146 | – |

The longest real test questions (152 and 153 characters) go through generation with prompts of 326–353 tokens. The 981-character input joins several English test questions; a dash under prompt tokens means it was stopped by the NOT_FOUND gate before generation, so its row measures preprocessing, retrieval and reranking of a maximum-length input.

## Latency against top-k

first 100 questions of data/test.jsonl shuffled with seed 2026, in process. The abstention threshold and minimum context score stay fixed, so quality columns show the trade-off, not a tuned optimum.

**Retrieved top-k** (fused candidates sent to the reranker; 4 passages to the generator):

| Retrieved top-k | Rerank p50 | Rerank p95 | Total p50 | Total p95 | Relevant chunk in context | Answered |
|---|---|---|---|---|---|---|
| 10 | 25.3 | 32.4 | 762 | 1,209 | 0.874 | 0.770 |
| 20 | 34.6 | 42.6 | 775 | 1,208 | 0.943 | 0.830 |
| 30 | 49.0 | 63.6 | 785 | 1,237 | 0.966 | 0.850 |
| 50 | 81.9 | 106 | 829 | 1,275 | 0.966 | 0.850 |

**Reranked top-k** (maximum passages sent to the generator; 30 candidates reranked):

| Reranked top-k | Prompt tokens avg | First token p50 | Generation p50 | Total p50 | Total p95 | Relevant chunk in context | Answered |
|---|---|---|---|---|---|---|---|
| 1 | 245.0 | 168 | 715 | 791 | 1,257 | 0.954 | 0.840 |
| 2 | 255.7 | 163 | 715 | 799 | 1,230 | 0.966 | 0.850 |
| 4 | 256.6 | 164 | 716 | 795 | 1,239 | 0.966 | 0.850 |
| 8 | 256.6 | 168 | 717 | 799 | 1,242 | 0.966 | 0.850 |

**Quick Search result count:**

| top_k | p50 | p95 | max |
|---|---|---|---|
| 5 | 5.6 | 6.4 | 12.3 |
| 10 | 6.8 | 7.4 | 10.9 |
| 20 | 7.7 | 8.9 | 9.3 |
| 50 | 10.2 | 12.8 | 20.9 |

## Latency against corpus size

From the corpus-scaling run on the same model versions (`results/day5/day5_l4_benchmark_20261007-2337/scaling/scaling.json`): the 292 real chunks plus deterministic synthetic distractors; the first 100 test questions; quality is not measured on these indexes. The pipeline code changed afterwards only by streaming the generator response.

| Chunks | Qdrant search | BM25 p50 | Embed p50 | Dense p50 | Rerank p50 | Total p50 | Total p95 |
|---|---|---|---|---|---|---|---|
| 1,000 | exact | 21.1 | 60.5 | 14.9 | 48.0 | 917 | 1,371 |
| 10,000 | exact | 24.8 | 59.4 | 16.1 | 46.4 | 920 | 1,536 |
| 100,000 | hnsw | 27.6 | 59.6 | 16.5 | 45.5 | 922 | 1,586 |

## Quick Search on a CPU-only host

**Hardware:** CPU AMD EPYC 7763 64-Core Processor, 4 usable vCPUs, 15,994 MiB RAM; GPU none; Linux-6.8.0-1064-azure-x86_64-with-glibc2.36. Deployed with `docker compose --profile cpu up -d` (OpenSearch + API with `API_MODES=quick_search`, CPU-only PyTorch image, no GPU).

| Requests | avg | p50 | p95 | max |
|---|---|---|---|---|
| 272 | 9.7 | 8.8 | 15.4 | 34.6 |

| Stage | avg | p50 | p95 | max |
|---|---|---|---|---|
| preprocessing | 0.0 | 0.0 | 0.1 | 0.1 |
| search | 7.2 | 6.3 | 12.2 | 31.1 |
| total | 7.3 | 6.4 | 12.3 | 31.2 |
