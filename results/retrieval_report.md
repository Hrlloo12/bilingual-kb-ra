# Retrieval report

Held-out test split, 238 answerable questions, 292-chunk corpus. All selections (fusion weights, reranker, candidate pool) used the validation split only.

## All systems, overall

| System | R@1 | R@5 | R@10 | R@20 | Hit@1 | Hit@5 | Hit@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|---|---|---|
| BM25 only (Quick Search) | 0.487 | 0.733 | 0.773 | 0.802 | 0.546 | 0.790 | 0.824 | 0.666 | 0.664 |
| Dense only, base embedder | 0.689 | 0.954 | 0.983 | 0.996 | 0.760 | 0.962 | 0.983 | 0.847 | 0.871 |
| Dense only, fine-tuned embedder | 0.880 | 0.996 | 0.996 | 1.000 | 0.954 | 0.996 | 0.996 | 0.971 | 0.976 |
| Hybrid BM25 + fine-tuned dense (equal RRF) | 0.565 | 0.802 | 0.884 | 0.990 | 0.634 | 0.849 | 0.916 | 0.750 | 0.755 |
| Hybrid, weights tuned on validation | 0.880 | 0.996 | 0.996 | 1.000 | 0.954 | 0.996 | 0.996 | 0.971 | 0.976 |
| Hybrid + base reranker | 0.868 | 0.979 | 0.992 | 1.000 | 0.941 | 0.979 | 0.992 | 0.959 | 0.966 |
| Hybrid + fine-tuned reranker (deployed) | 0.901 | 1.000 | 1.000 | 1.000 | 0.975 | 1.000 | 1.000 | 0.986 | 0.990 |

## Recall@1 by question type

| System | Arabic (MSA) (n=68) | Arabic (Saudi dialect) (n=41) | English (n=36) | Mixed AR/EN (n=19) | AR question → EN source (n=28) | EN question → AR source (n=46) | Cross-lingual (n=93) |
|---|---|---|---|---|---|---|---|
| BM25 only (Quick Search) | 0.779 | 0.427 | 0.819 | 0.368 | 0.000 | 0.196 | 0.097 |
| Dense only, base embedder | 0.868 | 0.683 | 0.736 | 0.763 | 0.250 | 0.630 | 0.452 |
| Dense only, fine-tuned embedder | 0.882 | 0.829 | 0.833 | 0.868 | 0.821 | 1.000 | 0.914 |
| Hybrid BM25 + fine-tuned dense (equal RRF) | 0.824 | 0.476 | 0.819 | 0.447 | 0.000 | 0.457 | 0.237 |
| Hybrid, weights tuned on validation | 0.882 | 0.829 | 0.833 | 0.868 | 0.821 | 1.000 | 0.914 |
| Hybrid + base reranker | 0.882 | 0.805 | 0.833 | 0.921 | 0.821 | 0.935 | 0.892 |
| Hybrid + fine-tuned reranker (deployed) | 0.897 | 0.854 | 0.833 | 0.921 | 0.893 | 1.000 | 0.957 |

## Recall@5 by question type

| System | Arabic (MSA) (n=68) | Arabic (Saudi dialect) (n=41) | English (n=36) | Mixed AR/EN (n=19) | AR question → EN source (n=28) | EN question → AR source (n=46) | Cross-lingual (n=93) |
|---|---|---|---|---|---|---|---|
| BM25 only (Quick Search) | 0.897 | 0.707 | 0.875 | 0.789 | 0.214 | 0.696 | 0.484 |
| Dense only, base embedder | 0.971 | 0.951 | 1.000 | 1.000 | 0.893 | 0.913 | 0.914 |
| Dense only, fine-tuned embedder | 1.000 | 1.000 | 1.000 | 1.000 | 0.964 | 1.000 | 0.989 |
| Hybrid BM25 + fine-tuned dense (equal RRF) | 0.919 | 0.817 | 0.889 | 1.000 | 0.357 | 0.739 | 0.613 |
| Hybrid, weights tuned on validation | 1.000 | 1.000 | 1.000 | 1.000 | 0.964 | 1.000 | 0.989 |
| Hybrid + base reranker | 1.000 | 1.000 | 1.000 | 1.000 | 0.893 | 0.957 | 0.946 |
| Hybrid + fine-tuned reranker (deployed) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Recall@10 by question type

| System | Arabic (MSA) (n=68) | Arabic (Saudi dialect) (n=41) | English (n=36) | Mixed AR/EN (n=19) | AR question → EN source (n=28) | EN question → AR source (n=46) | Cross-lingual (n=93) |
|---|---|---|---|---|---|---|---|
| BM25 only (Quick Search) | 0.912 | 0.756 | 0.889 | 0.895 | 0.357 | 0.696 | 0.559 |
| Dense only, base embedder | 1.000 | 1.000 | 1.000 | 1.000 | 0.893 | 0.978 | 0.957 |
| Dense only, fine-tuned embedder | 1.000 | 1.000 | 1.000 | 1.000 | 0.964 | 1.000 | 0.989 |
| Hybrid BM25 + fine-tuned dense (equal RRF) | 0.941 | 0.927 | 0.931 | 1.000 | 0.607 | 0.848 | 0.785 |
| Hybrid, weights tuned on validation | 1.000 | 1.000 | 1.000 | 1.000 | 0.964 | 1.000 | 0.989 |
| Hybrid + base reranker | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.957 | 0.979 |
| Hybrid + fine-tuned reranker (deployed) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Recall@20 by question type

| System | Arabic (MSA) (n=68) | Arabic (Saudi dialect) (n=41) | English (n=36) | Mixed AR/EN (n=19) | AR question → EN source (n=28) | EN question → AR source (n=46) | Cross-lingual (n=93) |
|---|---|---|---|---|---|---|---|
| BM25 only (Quick Search) | 0.919 | 0.817 | 0.889 | 0.947 | 0.393 | 0.739 | 0.613 |
| Dense only, base embedder | 1.000 | 1.000 | 1.000 | 1.000 | 0.964 | 1.000 | 0.989 |
| Dense only, fine-tuned embedder | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Hybrid BM25 + fine-tuned dense (equal RRF) | 0.993 | 1.000 | 1.000 | 1.000 | 0.929 | 1.000 | 0.979 |
| Hybrid, weights tuned on validation | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Hybrid + base reranker | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Hybrid + fine-tuned reranker (deployed) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Hit@1 by question type

| System | Arabic (MSA) (n=68) | Arabic (Saudi dialect) (n=41) | English (n=36) | Mixed AR/EN (n=19) | AR question → EN source (n=28) | EN question → AR source (n=46) | Cross-lingual (n=93) |
|---|---|---|---|---|---|---|---|
| BM25 only (Quick Search) | 0.868 | 0.463 | 0.972 | 0.421 | 0.000 | 0.196 | 0.097 |
| Dense only, base embedder | 0.971 | 0.756 | 0.889 | 0.842 | 0.250 | 0.630 | 0.452 |
| Dense only, fine-tuned embedder | 0.985 | 0.902 | 1.000 | 0.947 | 0.821 | 1.000 | 0.914 |
| Hybrid BM25 + fine-tuned dense (equal RRF) | 0.926 | 0.537 | 0.972 | 0.526 | 0.000 | 0.457 | 0.237 |
| Hybrid, weights tuned on validation | 0.985 | 0.902 | 1.000 | 0.947 | 0.821 | 1.000 | 0.914 |
| Hybrid + base reranker | 0.985 | 0.878 | 1.000 | 1.000 | 0.821 | 0.935 | 0.892 |
| Hybrid + fine-tuned reranker (deployed) | 1.000 | 0.927 | 1.000 | 1.000 | 0.893 | 1.000 | 0.957 |

## Hit@5 by question type

| System | Arabic (MSA) (n=68) | Arabic (Saudi dialect) (n=41) | English (n=36) | Mixed AR/EN (n=19) | AR question → EN source (n=28) | EN question → AR source (n=46) | Cross-lingual (n=93) |
|---|---|---|---|---|---|---|---|
| BM25 only (Quick Search) | 0.985 | 0.756 | 1.000 | 0.842 | 0.214 | 0.696 | 0.484 |
| Dense only, base embedder | 0.985 | 0.976 | 1.000 | 1.000 | 0.893 | 0.913 | 0.914 |
| Dense only, fine-tuned embedder | 1.000 | 1.000 | 1.000 | 1.000 | 0.964 | 1.000 | 0.989 |
| Hybrid BM25 + fine-tuned dense (equal RRF) | 1.000 | 0.854 | 1.000 | 1.000 | 0.357 | 0.739 | 0.613 |
| Hybrid, weights tuned on validation | 1.000 | 1.000 | 1.000 | 1.000 | 0.964 | 1.000 | 0.989 |
| Hybrid + base reranker | 1.000 | 1.000 | 1.000 | 1.000 | 0.893 | 0.957 | 0.946 |
| Hybrid + fine-tuned reranker (deployed) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Hit@10 by question type

| System | Arabic (MSA) (n=68) | Arabic (Saudi dialect) (n=41) | English (n=36) | Mixed AR/EN (n=19) | AR question → EN source (n=28) | EN question → AR source (n=46) | Cross-lingual (n=93) |
|---|---|---|---|---|---|---|---|
| BM25 only (Quick Search) | 1.000 | 0.805 | 1.000 | 0.895 | 0.357 | 0.696 | 0.559 |
| Dense only, base embedder | 1.000 | 1.000 | 1.000 | 1.000 | 0.893 | 0.978 | 0.957 |
| Dense only, fine-tuned embedder | 1.000 | 1.000 | 1.000 | 1.000 | 0.964 | 1.000 | 0.989 |
| Hybrid BM25 + fine-tuned dense (equal RRF) | 1.000 | 0.951 | 1.000 | 1.000 | 0.607 | 0.848 | 0.785 |
| Hybrid, weights tuned on validation | 1.000 | 1.000 | 1.000 | 1.000 | 0.964 | 1.000 | 0.989 |
| Hybrid + base reranker | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.957 | 0.979 |
| Hybrid + fine-tuned reranker (deployed) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## MRR by question type

| System | Arabic (MSA) (n=68) | Arabic (Saudi dialect) (n=41) | English (n=36) | Mixed AR/EN (n=19) | AR question → EN source (n=28) | EN question → AR source (n=46) | Cross-lingual (n=93) |
|---|---|---|---|---|---|---|---|
| BM25 only (Quick Search) | 0.929 | 0.615 | 0.986 | 0.628 | 0.130 | 0.413 | 0.285 |
| Dense only, base embedder | 0.977 | 0.842 | 0.935 | 0.903 | 0.500 | 0.779 | 0.651 |
| Dense only, fine-tuned embedder | 0.990 | 0.947 | 1.000 | 0.974 | 0.872 | 1.000 | 0.944 |
| Hybrid BM25 + fine-tuned dense (equal RRF) | 0.963 | 0.704 | 0.986 | 0.728 | 0.231 | 0.617 | 0.447 |
| Hybrid, weights tuned on validation | 0.990 | 0.947 | 1.000 | 0.974 | 0.872 | 1.000 | 0.944 |
| Hybrid + base reranker | 0.993 | 0.932 | 1.000 | 1.000 | 0.858 | 0.949 | 0.918 |
| Hybrid + fine-tuned reranker (deployed) | 1.000 | 0.963 | 1.000 | 1.000 | 0.934 | 1.000 | 0.975 |

## nDCG@10 by question type

| System | Arabic (MSA) (n=68) | Arabic (Saudi dialect) (n=41) | English (n=36) | Mixed AR/EN (n=19) | AR question → EN source (n=28) | EN question → AR source (n=46) | Cross-lingual (n=93) |
|---|---|---|---|---|---|---|---|
| BM25 only (Quick Search) | 0.880 | 0.627 | 0.898 | 0.672 | 0.183 | 0.483 | 0.350 |
| Dense only, base embedder | 0.962 | 0.864 | 0.949 | 0.928 | 0.595 | 0.828 | 0.726 |
| Dense only, fine-tuned embedder | 0.990 | 0.961 | 1.000 | 0.981 | 0.893 | 1.000 | 0.954 |
| Hybrid BM25 + fine-tuned dense (equal RRF) | 0.918 | 0.736 | 0.921 | 0.786 | 0.304 | 0.664 | 0.517 |
| Hybrid, weights tuned on validation | 0.990 | 0.961 | 1.000 | 0.981 | 0.893 | 1.000 | 0.954 |
| Hybrid + base reranker | 0.993 | 0.949 | 1.000 | 1.000 | 0.890 | 0.949 | 0.931 |
| Hybrid + fine-tuned reranker (deployed) | 1.000 | 0.973 | 1.000 | 1.000 | 0.951 | 1.000 | 0.981 |

## Deployed Smart AI Search stack, before → after fine-tuning, by question language

From the end-to-end run: the reranked top 10 of the full stack (BM25 + dense, RRF, reranker). Before uses the base embedder and base reranker.

| Group | n | R@1 | R@5 | R@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|
| Arabic questions | 137 | 0.839 → 0.883 | 0.985 → 1.000 | 0.985 → 1.000 | 0.939 → 0.976 | 0.949 → 0.982 |
| English questions | 82 | 0.890 → 0.927 | 0.976 → 1.000 | 0.988 → 1.000 | 0.971 → 1.000 | 0.975 → 1.000 |
| Mixed questions | 19 | 0.921 → 0.921 | 1.000 → 1.000 | 1.000 → 1.000 | 1.000 → 1.000 | 1.000 → 1.000 |
| Cross-lingual | 74 | 0.878 → 0.960 | 0.946 → 1.000 | 0.960 → 1.000 | 0.899 → 0.975 | 0.913 → 0.982 |
| All | 238 | 0.863 → 0.901 | 0.983 → 1.000 | 0.987 → 1.000 | 0.955 → 0.986 | 0.962 → 0.990 |

## Interactive AI Search, second turn (346 test conversations)

| System | Hit@1 | Hit@5 | MRR@10 |
|---|---|---|---|
| Interactive (gate + rewrite + memory), this run | 0.899 | 0.965 | 0.928 |
| No rewrite (raw follow-up), rewrite-selection run | 0.341 | 0.540 | 0.448 |
| Oracle (gold standalone question), rewrite-selection run | 0.972 | 1.000 | 0.981 |

The rewrite-selection run used the same frozen `v2` prompt and reached Hit@1 0.913. Between the two runs 30 of 346 rewrites were worded differently; 6 conversations lost and 1 gained the first rank. Nothing in the rewrite path changed, so this is decoding variation: FP8 temperature-0 decoding is not fully deterministic, and the suggested follow-up requests now share vLLM batches with the rewrites.

| Languages (turn 1 > turn 2) | n | Hit@1 | Hit@5 | MRR@10 |
|---|---|---|---|---|
| ar>ar | 104 | 0.875 | 0.952 | 0.913 |
| ar>en | 10 | 1.000 | 1.000 | 1.000 |
| ar>mixed | 59 | 0.864 | 0.983 | 0.910 |
| en>ar | 75 | 0.813 | 0.920 | 0.861 |
| en>en | 98 | 1.000 | 1.000 | 1.000 |
