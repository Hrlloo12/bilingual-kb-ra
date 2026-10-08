# Human evaluation

`human_eval_sheet.csv` holds 30 Smart AI Search answers from the final NVIDIA L4 test run (`results/final_l4_20261008/smart_search/final_test.jsonl`):
- 10 Arabic questions, 10 English questions, and 10 mixed or cross-lingual questions;
- 3 of the 30 are unanswerable questions, where the correct behaviour is NOT_FOUND.

`scripts/build_human_eval.py` sampled the items with seed 2026, stratified by question type. The ratings in `human_eval_sheet.csv` were given by the project author after reading every answer against its question, sources and reference. Nothing in this repository fills them in automatically. The summary is in `human_eval_summary.json`.

`human_eval_before_sheet.csv` (optional) has the same 30 questions answered by the base models: base embedder and base reranker, same generator. Rate it in the same way to fill the human rows of the before-vs-after table.

## How to rate

Open the file in Excel or LibreOffice (UTF-8). For each row, read:
- `question`;
- `system_answer`;
- `sources`, which names the cited chunks;
- `source_text`, the text of the cited chunks;
- `reference_answer`, the gold answer.

`answer_language` says whether the answer is Arabic (`ar`) or English (`en`), so Arabic and English fluency can be reported separately.

Fill in every column below with a whole number from 1 to 5, except where noted:

| Column | What to judge |
|---|---|
| `meaning_accuracy_1_to_5` | Is the answer factually correct and does it keep the meaning of the reference answer? For a NOT_FOUND row, give 5 if the question really cannot be answered from the knowledge bank, and 1 if it can. |
| `faithfulness_1_to_5` | Is everything in the answer supported by `source_text`? 5 means nothing was added beyond the sources. |
| `fluency_1_to_5` | Is the answer natural and grammatical in its own language? Leave it empty for NOT_FOUND rows. |
| `completeness_1_to_5` | Does the answer give everything the question asked for, for example both the price and the code, or all requested conditions? |
| `citation_quality_1_to_5` | Do the cited sources contain the answer, without irrelevant or missing citations? For a NOT_FOUND row with no citations, give 5. |
| `language_correct_yes_no` | `yes` or `no`: is the answer in the language of the question? Mixed questions may be answered in Arabic with English terms. |
| `notes` | Free text: anything notable, such as a wrong attribute, a partial answer or a translation issue. |

Scale: 1 unusable · 2 major issues · 3 understandable but imperfect · 4 good · 5 excellent.

Then run:

```bash
python scripts/summarize_human_eval.py
python scripts/summarize_human_eval.py --sheet results/human_eval/human_eval_before_sheet.csv
python scripts/build_reports.py
```

`summarize_human_eval.py` writes `human_eval_summary.json` (or `human_eval_before_summary.json`). It contains:
- the mean of each column, overall and per group;
- fluency for Arabic and English answers separately;
- a composition check (10 / 10 / 10);
- the list of unrated items.

It rejects any rating outside 1–5. `build_reports.py` then puts the human scores into `results/before_vs_after.md` and `results/summary.json`.
