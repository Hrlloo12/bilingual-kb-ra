# Error analysis

## Sources
- **Smart AI Search:** the final NVIDIA L4 run on the 272-question test split (`results/day5/day5_l4_benchmark_20261007-2337/generation/generation_test_records.jsonl`).
- **Interactive AI Search:** the frozen `v2` rewrite prompt on 346 test conversations (`results/interactive/day4_l4_interactive_v2_20261007-2302/`).

## Method
`scripts/error_analysis.py` flags candidates automatically (`error_examples.json`). Every flagged example was then read by hand, because several automatic rules over-count. The table separates real errors from false alarms.

## Summary

| Category | Auto-flagged | Real errors after review | What happened |
|---|---|---|---|
| Retrieval errors | 7 | 7 | 2 dialect price questions retrieved dimension chunks; 5 Khobar showroom questions never retrieved the Khobar chunk |
| Hallucination / unsupported claim | 2 | 2 | `test_00017` (entity confusion); `test_00242` (wrong negation) |
| Citation errors | 3 | 3 | The same 3 dialect price questions cite the dimension chunk they wrongly answered from |
| Incorrect prices | 5 | 3 | Dialect price questions answered with dimensions. The 2 VAT answers ("Yes, listed prices include VAT") are correct but omit the 15% |
| Incorrect numbers | 10 | 0 | All correct partial answers: width only when width was asked, "Two" spelled out, 5% without the 12-month validity |
| Incorrect dates / durations | 4 | 0 | Correct leave entitlements (21 / 30 days) without restating the 5-year condition |
| Names / entities | 0 | 1 | `test_00017`: a WhatsApp customer-service number was invented from the Abha showroom phone number |
| Negation | 1 | 1 | `test_00242`: "Does the Khobar showroom offer kitchen design?" was answered "No", but it offers a kitchen design studio |
| Language drift | 1 | 1 | `test_00217`: English question answered in Arabic (the source is Arabic) |
| Mixed-language issues | 0 | 1 | `test_00242` is a mixed question ("design kitchen") |
| NOT_FOUND, false refusals | 5 | 5 | All about the Khobar showroom (phone and service) |
| NOT_FOUND, missed | 1 | 1 | `test_00017` |
| AR→EN weakness | 4 | 4 | The Khobar facts exist only in English; Arabic questions about them fail |
| Interactive memory / rewrite | 23 | 23 | See below |

## Root causes

**1. The Khobar showroom is effectively unreachable from Arabic (6 of the 238 answerable test questions).**
- **Where the facts live:** the Khobar phone number and service appear only in the English store locator (`store_locator_en`).
- **Retrieval failure:** in 3 cases the reranker was confident (top score > 0.99) about a different showroom's passage (`store_locator_en#7`), and the generator correctly refused rather than guess. In 2 more the abstention gate refused because the top score was low (0.04 and 0.79).
- **Negation error:** when the chunk *was* retrieved for the mixed question `test_00242`, the generator read "design kitchen" against "kitchen design studio" and answered "No". That is a wrong negation, not a refusal.
- **Likely cause:** the Arabic name "الخبر" is also a common Arabic word ("the news"), and the English chunk spells it "Al Khobar". This is the clearest remaining cross-lingual retrieval gap.

**2. Saudi-dialect "how much" is read as size (3 questions).**
- "بكم …؟" and "كم ياخذ …؟" mean "how much does it cost", but retrieval and the reranker ranked the product's dimension chunk first.
- The generator then answered faithfully from the wrong chunk. The answer is supported by its citation, which is exactly why the LLM judge missed it, but it does not answer the question.

**3. Entity confusion on an unanswerable question (`test_00017`).**
- A request for a WhatsApp customer-service number cleared the abstention gate (reranker 0.883 > 0.8445) because the Abha showroom phone passage looked relevant.
- The generator relabelled that number as the WhatsApp number.
- The number post-check cannot catch this, because the number does appear in the passage.

**4. Interactive rewrite failures (23 of 286 test follow-ups not at rank 1 where the gold question would be).**
- **Wrong attribute in the opposite direction:** price questions followed by an Arabic "what about X?" are sometimes rewritten as dimension questions, for example "What is the price of the Cloud Foam mattress?" → "وماذا عن مرتبة Ortho Firm؟" → "ما أبعاد مرتبة Ortho Firm؟". The balanced prompt fixed most dimension→price errors, but not every case.
- **Transliteration:** "Hail" stays in Latin script inside an Arabic rewrite ("فرع Hail") instead of "حائل", which lowers retrieval rank.
- **Bare follow-up after an unrelated attribute:** "العنوان؟" after a question about Jazan's same-day pickup was rewritten to "ما هو العنوان؟" without the place name.

## Judge reliability

The Qwen3-8B judge reports faithfulness 0.996 and relevance 0.994. Measured against the known errors above, it is too lenient:
- it marked `test_00017` and the 3 dialect price errors as *supported* and *fully relevant*;
- it caught only `test_00242`;
- it marked one correct answer (`test_00267`) as *partial*.

Its scores are therefore an upper bound. The label-based metrics and the human ratings are the reliable evidence.
