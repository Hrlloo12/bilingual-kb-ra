# Generation report

Smart AI Search on the 272-question test split, NVIDIA L4, generator Qwen3-4B-Instruct-2507-FP8. Label-based metrics use the gold fact labels of each chunk; judge metrics use Qwen3-8B-FP8 (temperature 0, run after the serving stack was stopped).

## Overall, before vs after fine-tuning the retrieval models

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

## After fine-tuning, by question language

| Metric | Arabic questions | English questions | Mixed questions | Cross-lingual |
|---|---|---|---|---|
| Faithfulness: all claims supported (judge, upper bound) | 1.000 | 1.000 | 0.947 | 1.000 |
| Hallucination rate among answered (judge, lower bound) | 0.000 | 0.000 | 0.053 | 0.000 |
| Answer relevance (judge, upper bound) | 0.992 | 1.000 | 0.947 | 1.000 |
| Context precision (facts) | 0.916 | 0.939 | 0.983 | 0.907 |
| Context recall (facts) | 0.949 | 1.000 | 1.000 | 0.946 |
| Citation precision (facts) | 0.972 | 1.000 | 0.983 | 1.000 |
| Citation recall (facts) | 0.977 | 1.000 | 1.000 | 1.000 |
| Answer in the expected language | 1.000 | 0.976 | 1.000 | 0.971 |
| Unanswerable → NOT_FOUND | 0.923 | 1.000 | 1.000 | – |
| False NOT_FOUND on answerable | 0.036 | 0.000 | 0.000 | 0.054 |
| BLEU vs reference | 54.8 | 53.0 | 43.6 | 52.4 |
| chrF vs reference | 66.8 | 66.2 | 51.7 | 65.2 |
| ROUGE-L vs reference | 0.708 | 0.655 | 0.673 | 0.660 |

## Interactive AI Search (second turns of 346 test conversations)

| Metric | Value |
|---|---|
| Answered | 0.893 |
| Cites a relevant chunk | 0.867 |
| Reference numbers all in the answer (answered) | 0.722 |
| Answer in the expected language | 0.987 |
| Follow-ups sent to rewrite (gate recall) | 1.000 |
| Standalone questions rewritten unnecessarily | 0.017 |
| Rewrites kept in the user's language | 0.990 |
| Turns with suggested follow-ups | 0.988 |
| Suggested follow-ups per turn (avg) | 1.98 |
| Suggested follow-ups in the user's language | 0.991 |

Judge caveat: checked against the errors found by hand, the judge marked faithful-but-wrong answers (an answer copied from a wrongly retrieved chunk) as supported. Its faithfulness and relevance figures are therefore upper bounds; label-based metrics and the human ratings are the reliable evidence.
