# Human evaluation

`human_eval_sheet.csv` holds 30 Smart AI Search answers from the final NVIDIA L4 test run (`results/day5/`). It has 10 Arabic, 10 English and 10 mixed or cross-lingual questions, including 2 unanswerable ones. The items were sampled with seed 2026 by `scripts/build_human_eval.py`.

The ratings must be given by a person. The rating columns are empty, and nothing in this repository fills them in automatically.

## How to rate

Open the file in Excel or LibreOffice (UTF-8), read `question`, `system_answer`, `sources`, `source_text` and `reference_answer`, then fill in:

| Column | Scale | Meaning |
|---|---|---|
| `correctness_1_to_5` | 1–5 | Is the answer factually correct for the question? For a NOT_FOUND row, give 5 if the question really cannot be answered from the knowledge bank. |
| `faithful_to_sources_1_to_5` | 1–5 | Is everything in the answer supported by `source_text`? 5 means nothing extra was added. |
| `citations_correct_yes_no` | yes / no | Do the cited sources actually contain the answer? |
| `language_correct_yes_no` | yes / no | Is the answer in the language of the question? Mixed questions may be answered in Arabic with English terms. |
| `fluency_1_to_5` | 1–5 | Is the answer natural and readable in its language? |
| `notes` | free text | Anything notable, such as a wrong attribute, a partial answer or a translation issue. |

Then run:

```bash
python scripts/summarize_human_eval.py
```

It writes `human_eval_summary.json` with means per column and per group, and lists any unrated items. The results can also be compared with the LLM judge verdicts in `results/day5/.../generation/generation_test_records.jsonl` to check how often the judge agrees with a person.
