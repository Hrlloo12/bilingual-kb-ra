from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag.config import REPO_ROOT
from rag.evaluation.metrics import aggregate, query_metrics

RESULTS = REPO_ROOT / "results"
INTERACTIVE_DAY4_DIR = RESULTS / "interactive" / "day4_l4_interactive_v2_20261007-2302" / "day4_l4_interactive_v2_20261007-2302"
CROSS_LINGUAL = {"ar_en", "en_ar"}
MODES = ("quick_search", "smart_ai_search", "interactive")
MODE_NAMES = {"quick_search": "Quick Search", "smart_ai_search": "Smart AI Search", "interactive": "Interactive AI Search"}
BUCKETS = ("ar_ar", "dialect", "en_en", "mixed", "ar_en", "en_ar")
BUCKET_NAMES = {
    "ar_ar": "Arabic (MSA)",
    "dialect": "Arabic (Saudi dialect)",
    "en_en": "English",
    "mixed": "Mixed AR/EN",
    "ar_en": "AR question → EN source",
    "en_ar": "EN question → AR source",
}
RETRIEVAL_SYSTEMS = (
    ("fusion", "bm25", "BM25 only (Quick Search)"),
    ("fusion", "dense_base", "Dense only, base embedder"),
    ("fusion", "dense_ft", "Dense only, fine-tuned embedder"),
    ("fusion", "hybrid_equal", "Hybrid BM25 + fine-tuned dense (equal RRF)"),
    ("fusion", "hybrid_tuned", "Hybrid, weights tuned on validation"),
    ("reranker", "hybrid_equal+rerank_base", "Hybrid + base reranker"),
    ("reranker", "hybrid_equal+rerank_finetuned", "Hybrid + fine-tuned reranker (deployed)"),
)
STAGE_ORDER = (
    "memory_read", "query_rewrite", "preprocessing", "search", "bm25", "embedding", "dense", "fusion", "retrieval", "rerank",
    "generation_first_token", "time_to_first_token", "generation_total", "post_checks", "followups", "followups_wait",
    "smart_search", "memory_write", "total",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def ms(value) -> str:
    return "–" if value is None else f"{value:,.0f}" if value >= 100 else f"{value:,.1f}"


def table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def stack_retrieval(records: list[dict]) -> dict:
    groups = {"ar": [], "en": [], "mixed": [], "cross_lingual": [], "all": []}
    for record in records:
        if not record["answerable"] or "ranked_chunk_ids" not in record:
            continue
        metrics = query_metrics(record["ranked_chunk_ids"], set(record["relevant_chunk_ids"]))
        groups["all"].append(metrics)
        groups[record["language"]].append(metrics)
        if record["bucket"] in CROSS_LINGUAL:
            groups["cross_lingual"].append(metrics)
    return {name: {"n": len(rows), **aggregate(rows)} for name, rows in groups.items()}


def latency(summary: dict, stage: str, statistic: str):
    return summary.get("latency_ms", {}).get(stage, {}).get(statistic)


class Inputs:
    def __init__(self, run: Path, cpu: Path, scaling: Path) -> None:
        self.run = run
        self.after_summary = load(run / "smart_search" / "final_summary.json")
        self.before_summary = load(run / "before" / "before_summary.json")
        self.after_records = read_jsonl(run / "smart_search" / "final_test.jsonl")
        self.before_records = read_jsonl(run / "before" / "before_test.jsonl")
        self.after_generation = load(run / "generation" / "after_test.json")
        self.before_generation = load(run / "generation" / "before_test.json")
        self.before_threshold = load(run / "before" / "abstention_end_to_end.json")
        self.api_latency = load(run / "latency" / "api.json")
        self.topk = load(run / "latency" / "topk.json")
        self.concurrency = {mode: load(run / "concurrency" / f"{mode}.json") for mode in MODES}
        self.interactive = load(run / "interactive" / "summary.json")
        self.smoke = load(run / "smoke.json")
        self.cpu_latency = load(cpu / "latency_api.json")
        self.cpu_concurrency = load(cpu / "concurrency_quick_search.json")
        self.scaling = load(scaling)
        self.fusion = load(RESULTS / "fusion" / "retrieval_comparison_test.json")["systems"]
        self.reranker = load(RESULTS / "reranker" / "reranker_comparison_test.json")["systems"]
        self.interactive_day4 = load(INTERACTIVE_DAY4_DIR / "summary.json")
        human = RESULTS / "human_eval"
        self.human_after = load(human / "human_eval_summary.json") if (human / "human_eval_summary.json").exists() else None
        self.human_before = load(human / "human_eval_before_summary.json") if (human / "human_eval_before_summary.json").exists() else None

    def system(self, source: str, name: str) -> dict:
        return (self.fusion if source == "fusion" else self.reranker)[name]


def human_value(summary: dict | None, key: str):
    if not summary:
        return None
    if key in ("ar", "en"):
        return summary["fluency_by_answer_language"].get(key)
    return summary["overall"][key]["mean"]


def before_after_rows(data: Inputs) -> list[dict]:
    before_retrieval = stack_retrieval(data.before_records)
    after_retrieval = stack_retrieval(data.after_records)
    before_answers = data.before_summary["splits"]["test"]["overall"]
    after_answers = data.after_summary["splits"]["test"]["overall"]
    before_gen = data.before_generation["overall"]
    after_gen = data.after_generation["overall"]
    quick_p95 = data.api_latency["modes"]["quick_search"]["client_total_ms"]["p95"]
    rows = []

    def add(metric: str, before, after, kind: str = "rate") -> None:
        render = {"rate": lambda value: fmt(value), "ms": ms, "mib": lambda value: "–" if value is None else f"{value:,.0f}", "score": lambda value: fmt(value, 2)}[kind]
        rows.append({"metric": metric, "before": render(before), "after": render(after), "raw_before": before, "raw_after": after})

    for language, label in (("ar", "AR"), ("en", "EN"), ("mixed", "mixed"), ("cross_lingual", "cross-lingual")):
        add(f"Recall@5 ({label})", before_retrieval[language].get("recall@5"), after_retrieval[language].get("recall@5"))
    for metric, key in (("MRR", "mrr"), ("nDCG@10", "ndcg@10")):
        for language, label in (("ar", "AR"), ("en", "EN")):
            add(f"{metric} ({label})", before_retrieval[language].get(key), after_retrieval[language].get(key))
    add("Faithfulness (LLM judge, upper bound)", before_gen["faithfulness_supported_rate"], after_gen["faithfulness_supported_rate"])
    add("Answer relevance (LLM judge, upper bound)", before_gen["answer_relevance"], after_gen["answer_relevance"])
    add("Hallucination rate among answered (LLM judge, lower bound)", before_gen["hallucination_rate_answered"], after_gen["hallucination_rate_answered"])
    add("Answerable questions answered", before_answers["answerable_answered_rate"], after_answers["answerable_answered_rate"])
    add("Unanswerable questions → NOT_FOUND", before_answers["unanswerable_not_found_rate"], after_answers["unanswerable_not_found_rate"])
    add("Citation precision (facts)", before_gen["citation_precision"], after_gen["citation_precision"])
    add("Human meaning score (1–5)", human_value(data.human_before, "meaning_accuracy_1_to_5"), human_value(data.human_after, "meaning_accuracy_1_to_5"), "score")
    add("Human fluency score, Arabic answers (1–5)", human_value(data.human_before, "ar"), human_value(data.human_after, "ar"), "score")
    add("Human fluency score, English answers (1–5)", human_value(data.human_before, "en"), human_value(data.human_after, "en"), "score")
    add("Time to first token, avg (ms)", latency(before_answers, "time_to_first_token", "avg"), latency(after_answers, "time_to_first_token", "avg"), "ms")
    add("Time to first token, p95 (ms)", latency(before_answers, "time_to_first_token", "p95"), latency(after_answers, "time_to_first_token", "p95"), "ms")
    add("Total latency, Smart AI, avg (ms)", latency(before_answers, "total", "avg"), latency(after_answers, "total", "avg"), "ms")
    add("Total latency, Smart AI, p95 (ms)", latency(before_answers, "total", "p95"), latency(after_answers, "total", "p95"), "ms")
    add("Quick Search p95 latency (ms, no fine-tuned component)", quick_p95, quick_p95, "ms")
    add("GPU memory used, whole stack (MiB)", max(data.before_summary["gpu_memory_used_mib_after_run"] or [0]) or None, max(data.after_summary["gpu_memory_used_mib_after_run"] or [0]) or None, "mib")
    add(
        "GPU memory, embedder + reranker peak allocated (MiB)",
        data.before_summary.get("torch_peak_allocated_mib_embedder_reranker"),
        data.after_summary.get("torch_peak_allocated_mib_embedder_reranker"),
        "mib",
    )
    before_memory = data.before_summary.get("memory_mib_after_run", {})
    after_memory = data.after_summary.get("memory_mib_after_run", {})
    add("CPU memory, host RAM used (MiB)", before_memory.get("host_ram_used"), after_memory.get("host_ram_used"), "mib")
    add("CPU memory, retrieval process RSS (MiB)", before_memory.get("evaluation_process_rss"), after_memory.get("evaluation_process_rss"), "mib")
    return rows


def before_after_markdown(data: Inputs, rows: list[dict]) -> str:
    return "\n\n".join(
        [
            "# Before vs after fine-tuning",
            "Both columns ran the same pipeline in the same NVIDIA L4 job: equal-weight RRF over BM25 and dense retrieval (30 candidates), "
            "cross-encoder reranking, a top-score NOT_FOUND gate, and Qwen3-4B-Instruct-2507-FP8 on vLLM with streaming. "
            "Only the two retrieval models change.",
            table(
                ["", "Before", "After"],
                [
                    ["Embedder", "Qwen/Qwen3-Embedding-0.6B", "halarash/qimam-qwen3-embedding-0.6b-kb-v2"],
                    ["Reranker", "BAAI/bge-reranker-v2-m3", "halarash/qimam-bge-reranker-v2-m3-kb"],
                    [
                        "NOT_FOUND gate threshold (validation-calibrated)",
                        fmt(data.before_summary["config"]["smart_search"]["abstain_threshold"], 4),
                        fmt(data.after_summary["config"]["smart_search"]["abstain_threshold"], 4),
                    ],
                ],
            ),
            "Test split: 272 questions (238 answerable, 34 unanswerable). Retrieval rows use the deployed stack's reranked top 10 for each "
            "answerable question, grouped by question language; cross-lingual means the fact exists only in the other language. "
            "Latency rows are sequential, one request at a time, measured in process (no HTTP).",
            table(["Metric", "Before fine-tuning", "After fine-tuning"], [[row["metric"], row["before"], row["after"]] for row in rows]),
            "Human scores come from `results/human_eval/`; a dash means not rated by a person. "
            "The LLM-judge figures are bounds: on the known errors the Qwen3-8B judge is lenient (see `results/error_analysis.md`).",
        ]
    ) + "\n"


def retrieval_markdown(data: Inputs) -> str:
    parts = ["# Retrieval report", "Held-out test split, 238 answerable questions, 292-chunk corpus. All selections (fusion weights, reranker, candidate pool) used the validation split only."]
    overall_rows = []
    for source, name, label in RETRIEVAL_SYSTEMS:
        overall = data.system(source, name)["overall"]
        overall_rows.append([label, *(fmt(overall[key]) for key in ("recall@1", "recall@5", "recall@10", "recall@20", "hit@1", "hit@5", "hit@10", "mrr", "ndcg@10"))])
    parts += ["## All systems, overall", table(["System", "R@1", "R@5", "R@10", "R@20", "Hit@1", "Hit@5", "Hit@10", "MRR", "nDCG@10"], overall_rows)]
    per_type = (
        ("recall@1", "Recall@1"), ("recall@5", "Recall@5"), ("recall@10", "Recall@10"), ("recall@20", "Recall@20"),
        ("hit@1", "Hit@1"), ("hit@5", "Hit@5"), ("hit@10", "Hit@10"), ("mrr", "MRR"), ("ndcg@10", "nDCG@10"),
    )
    for metric, label in per_type:
        rows = []
        for source, name, system_label in RETRIEVAL_SYSTEMS:
            system = data.system(source, name)
            rows.append([system_label, *(fmt(system["by_bucket"][bucket][metric]) for bucket in BUCKETS), fmt(system["cross_lingual"][metric])])
        headers = ["System", *(f"{BUCKET_NAMES[bucket]} (n={data.system('fusion', 'bm25')['by_bucket'][bucket]['n']})" for bucket in BUCKETS), "Cross-lingual (n=93)"]
        parts += [f"## {label} by question type", table(headers, rows)]

    before, after = stack_retrieval(data.before_records), stack_retrieval(data.after_records)
    rows = []
    for group, label in (("ar", "Arabic questions"), ("en", "English questions"), ("mixed", "Mixed questions"), ("cross_lingual", "Cross-lingual"), ("all", "All")):
        rows.append([label, after[group]["n"], *(f"{fmt(before[group].get(key))} → {fmt(after[group].get(key))}" for key in ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"))])
    parts += [
        "## Deployed Smart AI Search stack, before → after fine-tuning, by question language",
        "From the end-to-end run: the reranked top 10 of the full stack (BM25 + dense, RRF, reranker). Before uses the base embedder and base reranker.",
        table(["Group", "n", "R@1", "R@5", "R@10", "MRR@10", "nDCG@10"], rows),
    ]

    run = data.interactive["splits"]["test"]
    day4 = data.interactive_day4["splits"]["test"]
    rows = [
        ["Interactive (gate + rewrite + memory), this run", fmt(run["overall"]["systems"]["interactive"]["hit@1"]), fmt(run["overall"]["systems"]["interactive"]["hit@5"]), fmt(run["overall"]["systems"]["interactive"]["mrr@10"])],
        ["No rewrite (raw follow-up), rewrite-selection run", fmt(day4["overall"]["systems"]["no_rewrite"]["hit@1"]), fmt(day4["overall"]["systems"]["no_rewrite"]["hit@5"]), fmt(day4["overall"]["systems"]["no_rewrite"]["mrr@10"])],
        ["Oracle (gold standalone question), rewrite-selection run", fmt(day4["overall"]["systems"]["oracle"]["hit@1"]), fmt(day4["overall"]["systems"]["oracle"]["hit@5"]), fmt(day4["overall"]["systems"]["oracle"]["mrr@10"])],
    ]
    new = {record["id"]: record for record in read_jsonl(data.run / "interactive" / "records_test.jsonl")}
    old = {record["id"]: record for record in read_jsonl(INTERACTIVE_DAY4_DIR / "records_test.jsonl")}
    reworded = sum(1 for key in new if new[key]["rewrite"]["standalone_query"] != old[key]["rewrite"]["standalone_query"])
    lost = sum(1 for key in new if old[key]["interactive"]["rank"] == 1 and new[key]["interactive"]["rank"] != 1)
    gained = sum(1 for key in new if old[key]["interactive"]["rank"] != 1 and new[key]["interactive"]["rank"] == 1)
    parts += [
        "## Interactive AI Search, second turn (346 test conversations)",
        table(["System", "Hit@1", "Hit@5", "MRR@10"], rows),
        f"The rewrite-selection run used the same frozen `v2` prompt and reached Hit@1 {fmt(day4['overall']['systems']['interactive']['hit@1'])}. "
        f"Between the two runs {reworded} of {len(new)} rewrites were worded differently; {lost} conversations lost and {gained} gained the first rank. "
        "Nothing in the rewrite path changed, so this is decoding variation: FP8 temperature-0 decoding is not fully deterministic, and the suggested follow-up requests now share vLLM batches with the rewrites.",
        table(
            ["Languages (turn 1 > turn 2)", "n", "Hit@1", "Hit@5", "MRR@10"],
            [
                [name, summary["systems"]["interactive"]["n"], fmt(summary["systems"]["interactive"]["hit@1"]), fmt(summary["systems"]["interactive"]["hit@5"]), fmt(summary["systems"]["interactive"]["mrr@10"])]
                for name, summary in run["by_languages"].items()
            ],
        ),
    ]
    return "\n\n".join(parts) + "\n"


def generation_markdown(data: Inputs) -> str:
    keys = (
        ("faithfulness_supported_rate", "Faithfulness: all claims supported (judge, upper bound)"),
        ("hallucination_rate_answered", "Hallucination rate among answered (judge, lower bound)"),
        ("answer_relevance", "Answer relevance (judge, upper bound)"),
        ("context_precision", "Context precision (facts)"),
        ("context_recall_facts", "Context recall (facts)"),
        ("citation_precision", "Citation precision (facts)"),
        ("citation_recall_facts", "Citation recall (facts)"),
        ("language_correct", "Answer in the expected language"),
        ("not_found_accuracy_unanswerable", "Unanswerable → NOT_FOUND"),
        ("false_not_found_rate", "False NOT_FOUND on answerable"),
        ("bleu", "BLEU vs reference"),
        ("chrf", "chrF vs reference"),
        ("rouge_l", "ROUGE-L vs reference"),
    )
    before, after = data.before_generation, data.after_generation
    parts = [
        "# Generation report",
        "Smart AI Search on the 272-question test split, NVIDIA L4, generator Qwen3-4B-Instruct-2507-FP8. "
        "Label-based metrics use the gold fact labels of each chunk; judge metrics use Qwen3-8B-FP8 (temperature 0, run after the serving stack was stopped).",
        "## Overall, before vs after fine-tuning the retrieval models",
        table(["Metric", "Before", "After"], [[label, fmt(before["overall"].get(key), 3 if key not in ("bleu", "chrf") else 1), fmt(after["overall"].get(key), 3 if key not in ("bleu", "chrf") else 1)] for key, label in keys]),
    ]
    groups = [("ar", "Arabic questions"), ("en", "English questions"), ("mixed", "Mixed questions")]
    rows = []
    for key, label in keys:
        row = [label]
        for group, _ in groups:
            row.append(fmt(after["by_language"].get(group, {}).get(key), 3 if key not in ("bleu", "chrf") else 1))
        row.append(fmt(after["cross_lingual"].get(key), 3 if key not in ("bleu", "chrf") else 1))
        rows.append(row)
    parts += ["## After fine-tuning, by question language", table(["Metric", *(label for _, label in groups), "Cross-lingual"], rows)]

    interactive = data.interactive["splits"]["test"]["overall"]
    system = interactive["systems"]["interactive"]
    followups = interactive.get("suggested_followups", {})
    parts += [
        "## Interactive AI Search (second turns of 346 test conversations)",
        table(
            ["Metric", "Value"],
            [
                ["Answered", fmt(system["answered"])],
                ["Cites a relevant chunk", fmt(system["cited_relevant"])],
                ["Reference numbers all in the answer (answered)", fmt(system["numbers_ok"])],
                ["Answer in the expected language", fmt(system["language_ok"])],
                ["Follow-ups sent to rewrite (gate recall)", fmt(interactive["gate"]["follow_up_recall"])],
                ["Standalone questions rewritten unnecessarily", fmt(interactive["gate"]["standalone_rewritten_rate"])],
                ["Rewrites kept in the user's language", fmt(interactive["rewrite"]["language_preserved"])],
                ["Turns with suggested follow-ups", fmt(followups.get("with_suggestions"))],
                ["Suggested follow-ups per turn (avg)", fmt(followups.get("average_count"), 2)],
                ["Suggested follow-ups in the user's language", fmt(followups.get("in_user_language"))],
            ],
        ),
        "Judge caveat: checked against the errors found by hand, the judge marked faithful-but-wrong answers (an answer copied from a wrongly retrieved chunk) as supported. "
        "Its faithfulness and relevance figures are therefore upper bounds; label-based metrics and the human ratings are the reliable evidence.",
    ]
    return "\n\n".join(parts) + "\n"


def stage_table(stages: dict) -> str:
    rows = [[stage, ms(values["avg"]), ms(values["p50"]), ms(values["p95"]), ms(values["max"])] for stage in STAGE_ORDER if (values := stages.get(stage))]
    return table(["Stage", "avg", "p50", "p95", "max"], rows)


def host_line(host: dict) -> str:
    gpus = ", ".join(f"{gpu['name']} ({int(float(gpu['memory_total_mib'])):,} MiB, driver {gpu['driver']})" for gpu in host.get("gpus", [])) or "none"
    return f"CPU {host['cpu_model']}, {host['usable_cpus']} usable vCPUs, {host['ram_total_mib']:,} MiB RAM; GPU {gpus}; {host['os']}."


def latency_markdown(data: Inputs) -> str:
    api = data.api_latency
    parts = [
        "# Latency report",
        f"**Hardware (GPU run):** {host_line(api['host'])}",
        "Requests were sent one at a time through the HTTP API (`scripts/benchmark_latency.py api`). Client time includes HTTP, JSON serialization and the full pipeline; "
        "server stages come from each response's `latency_ms`. Time to first token is measured server-side from the moment the request is received until vLLM streams "
        "the first answer token; the API returns the answer only after the number/citation post-checks, so it does not stream tokens to the client.",
    ]
    rows = []
    for mode in MODES:
        report = api["modes"][mode]
        client = report["client_total_ms"]
        first_result = report.get("time_to_first_result_ms", {})
        first_token = report.get("time_to_first_token_ms", {})
        rows.append([MODE_NAMES[mode], report["requests"], ms(client["avg"]), ms(client["p50"]), ms(client["p95"]), ms(client["max"]), ms(first_result.get("p50")), ms(first_result.get("p95")), ms(first_token.get("p50")), ms(first_token.get("p95"))])
    parts += [
        "## End to end, per mode (ms)",
        table(["Mode", "Requests", "avg", "p50", "p95", "max", "First result p50", "First result p95", "First token p50", "First token p95"], rows),
        "Targets: Quick Search p95 < 100 ms; Smart AI Search first token < 1,500 ms and total < 4,000 ms at p95; Interactive rewrite < 200 ms.",
    ]
    for mode in MODES:
        parts += [f"## {MODE_NAMES[mode]}: stages (server, ms)", stage_table(api["modes"][mode]["server_stages_ms"])]
        if mode == "interactive":
            follow = api["modes"][mode]["follow_up_turns"]
            parts += ["Follow-up turns only (turn 2, where the rewrite runs):", stage_table(follow["server_stages_ms"])]

    smart = data.after_summary["splits"]["test"]["overall"]
    parts += ["## Smart AI Search in process, all 272 test questions (ms)", "No HTTP; queries stopped by the NOT_FOUND gate skip generation.", stage_table(smart["latency_ms"])]

    rows = []
    for name, entry in api["max_input"]["inputs"].items():
        for mode in MODES:
            report = entry["modes"][mode]
            rows.append([name.replace("_", " "), entry["chars"], MODE_NAMES[mode], ms(report["client_total_ms"]["avg"]), ms(report["client_total_ms"]["max"]), fmt(report.get("prompt_tokens", {}).get("max"))])
    parts += [
        "## Maximum input length tested",
        f"The API accepts up to {api['max_input']['api_limit_chars']:,} characters per question. Each input was sent {len(next(iter(api['max_input']['inputs'].values()))['modes'])} modes × 5 times.",
        table(["Input", "Characters", "Mode", "avg ms", "max ms", "Max prompt tokens"], rows),
        "The longest real test questions (152 and 153 characters) go through generation with prompts of 326–353 tokens. "
        "The 981-character input joins several English test questions; a dash under prompt tokens means it was stopped by the NOT_FOUND gate before generation, "
        "so its row measures preprocessing, retrieval and reranking of a maximum-length input.",
    ]

    topk = data.topk
    retrieved_rows = [[item["retrieved_top_k"], ms(latency({"latency_ms": item["latency_ms"]}, "rerank", "p50")), ms(latency({"latency_ms": item["latency_ms"]}, "rerank", "p95")), ms(latency({"latency_ms": item["latency_ms"]}, "total", "p50")), ms(latency({"latency_ms": item["latency_ms"]}, "total", "p95")), fmt(item["context_hit_rate_answerable"]), fmt(item["answered_rate"])] for item in topk["retrieved_sweep"]]
    reranked_rows = [[item["reranked_top_k"], fmt(item["prompt_tokens_avg"], 1), ms(latency({"latency_ms": item["latency_ms"]}, "time_to_first_token", "p50")), ms(latency({"latency_ms": item["latency_ms"]}, "generation_total", "p50")), ms(latency({"latency_ms": item["latency_ms"]}, "total", "p50")), ms(latency({"latency_ms": item["latency_ms"]}, "total", "p95")), fmt(item["context_hit_rate_answerable"]), fmt(item["answered_rate"])] for item in topk["reranked_sweep"]]
    quick_rows = [[item["top_k"], ms(item["latency_ms"]["total"]["p50"]), ms(item["latency_ms"]["total"]["p95"]), ms(item["latency_ms"]["total"]["max"])] for item in topk["quick_search_sweep"]]
    parts += [
        "## Latency against top-k",
        f"{topk['query_pool']}, in process. The abstention threshold and minimum context score stay fixed, so quality columns show the trade-off, not a tuned optimum.",
        "**Retrieved top-k** (fused candidates sent to the reranker; 4 passages to the generator):",
        table(["Retrieved top-k", "Rerank p50", "Rerank p95", "Total p50", "Total p95", "Relevant chunk in context", "Answered"], retrieved_rows),
        "**Reranked top-k** (maximum passages sent to the generator; 30 candidates reranked):",
        table(["Reranked top-k", "Prompt tokens avg", "First token p50", "Generation p50", "Total p50", "Total p95", "Relevant chunk in context", "Answered"], reranked_rows),
        "**Quick Search result count:**",
        table(["top_k", "p50", "p95", "max"], quick_rows),
    ]

    rows = []
    for size in data.scaling["sizes"]:
        stages = size["latency_ms"]
        rows.append([f"{size['chunks']:,}", size["qdrant"]["search"], ms(stages["bm25"]["p50"]), ms(stages["embedding"]["p50"]), ms(stages["dense"]["p50"]), ms(stages["rerank"]["p50"]), ms(stages["total"]["p50"]), ms(stages["total"]["p95"])])
    parts += [
        "## Latency against corpus size",
        "From the corpus-scaling run on the same model versions (`results/day5/day5_l4_benchmark_20261007-2337/scaling/scaling.json`): the 292 real chunks plus deterministic synthetic distractors; "
        "the first 100 test questions; quality is not measured on these indexes. The pipeline code changed afterwards only by streaming the generator response.",
        table(["Chunks", "Qdrant search", "BM25 p50", "Embed p50", "Dense p50", "Rerank p50", "Total p50", "Total p95"], rows),
    ]

    cpu = data.cpu_latency
    report = cpu["modes"]["quick_search"]
    parts += [
        "## Quick Search on a CPU-only host",
        f"**Hardware:** {host_line(cpu['host'])} Deployed with `docker compose --profile cpu up -d` (OpenSearch + API with `API_MODES=quick_search`, CPU-only PyTorch image, no GPU).",
        table(["Requests", "avg", "p50", "p95", "max"], [[report["requests"], ms(report["client_total_ms"]["avg"]), ms(report["client_total_ms"]["p50"]), ms(report["client_total_ms"]["p95"]), ms(report["client_total_ms"]["max"])]]),
        stage_table(report["server_stages_ms"]),
    ]
    return "\n\n".join(parts) + "\n"


def concurrency_rows(report: dict) -> list[list]:
    rows = []
    for level in report["levels"]:
        resources = level["resources"]
        rows.append(
            [
                level["users"], level["requests"], f"{level['failures']} ({level.get('failure_rate', 0):.1%})", fmt(level["qps"], 2),
                ms(level["latency_ms"]["avg"]), ms(level["latency_ms"]["p50"]), ms(level["latency_ms"]["p95"]), ms(level["latency_ms"]["max"]),
                fmt(resources.get("gpu_util_percent", {}).get("avg"), 0), fmt(resources.get("gpu_memory_mib", {}).get("max"), 0),
                fmt(resources.get("cpu_percent", {}).get("avg"), 0), fmt(resources.get("host_ram_used_mib", {}).get("max"), 0),
            ]
        )
    return rows


def concurrency_markdown(data: Inputs) -> str:
    headers = ["Users", "Requests", "Failures", "QPS", "avg ms", "p50", "p95", "max", "GPU util avg %", "GPU mem max MiB", "CPU avg %", "Host RAM max MiB"]
    parts = [
        "# Concurrency report",
        f"**Hardware:** {host_line(data.concurrency['smart_ai_search']['host'])}",
        "Closed-loop load: each simulated user sends its next request as soon as the previous one returns. Every level draws from the same shuffled question order (seed 2026). "
        "Interactive users run real two-turn conversations, each in its own session, so half of the interactive requests are follow-ups that go through the rewrite. "
        "GPU and host figures are sampled every second during the level.",
    ]
    for mode in MODES:
        report = data.concurrency[mode]
        parts += [f"## {MODE_NAMES[mode]}", table(headers, concurrency_rows(report))]
        if mode == "quick_search":
            parts.append(
                "On the GPU host, Quick Search levels with 1–8 users finished in under one second, before the first one-second resource sample, so only the 16-user level has resource figures. "
                "Quick Search does not use the GPU; the GPU memory shown is the idle serving stack. The CPU-only run below was measured after the sampler was changed to also sample at the start and end of each level."
            )
        if mode == "interactive":
            rows = [[level["users"], ms(level["follow_up_latency_ms"]["avg"]), ms(level["follow_up_latency_ms"]["p95"])] for level in report["levels"] if level.get("follow_up_latency_ms")]
            parts += ["Follow-up turns only:", table(["Users", "avg ms", "p95 ms"], rows)]
    parts += [
        "## Quick Search on a CPU-only host",
        f"**Hardware:** {host_line(data.cpu_concurrency['host'])}",
        table(headers, concurrency_rows(data.cpu_concurrency)),
    ]
    return "\n\n".join(parts) + "\n"


def summary_json(data: Inputs, rows: list[dict]) -> dict:
    latency_rows = []
    for mode in MODES:
        report = data.api_latency["modes"][mode]
        client = report["client_total_ms"]
        latency_rows.append(
            {"mode": MODE_NAMES[mode], "avg": round(client["avg"]), "p50": round(client["p50"]), "p95": round(client["p95"]), "max": round(client["max"]), "first_token_p95": round(report["time_to_first_token_ms"]["p95"]) if "time_to_first_token_ms" in report else "–"}
        )
    concurrency = []
    for index, level in enumerate(data.concurrency["smart_ai_search"]["levels"]):
        row = {"users": level["users"]}
        for mode in MODES:
            item = data.concurrency[mode]["levels"][index]
            row[mode] = f"{item['latency_ms']['p95']:.0f} ms / {item['qps']:.1f}"
        concurrency.append(row)
    return {"before_vs_after": [{key: row[key] for key in ("metric", "before", "after")} for row in rows], "latency": latency_rows, "concurrency": concurrency}


def update_metrics_files(data: Inputs) -> None:
    for stage, summary, records, generation in (
        ("before", data.before_summary, data.before_records, data.before_generation),
        ("after", data.after_summary, data.after_records, data.after_generation),
    ):
        path = RESULTS / f"{stage}_finetuning_metrics.json"
        content = load(path)
        content["end_to_end"] = {
            "source": str(data.run.relative_to(REPO_ROOT)).replace("\\", "/"),
            "models": {key: summary["config"][key] for key in ("embedding", "reranker", "generator")},
            "abstain_threshold": summary["config"]["smart_search"]["abstain_threshold"],
            "stack_retrieval_test_by_language": stack_retrieval(records),
            "answers_test": summary["splits"]["test"]["overall"],
            "answers_validation": summary["splits"]["validation"]["overall"],
            "generation_test": {"overall": generation["overall"], "by_language": generation["by_language"], "cross_lingual": generation["cross_lingual"]},
            "gpu_memory_used_mib_after_run": summary["gpu_memory_used_mib_after_run"],
            "torch_peak_allocated_mib_embedder_reranker": summary.get("torch_peak_allocated_mib_embedder_reranker"),
            "memory_mib_after_run": summary.get("memory_mib_after_run"),
            "host": summary.get("host"),
        }
        path.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the markdown reports and results/summary.json from measured result files.")
    parser.add_argument("--run", type=Path, default=RESULTS / "final_l4_20261008")
    parser.add_argument("--cpu", type=Path, default=RESULTS / "cpu_quick_search")
    parser.add_argument("--scaling", type=Path, default=RESULTS / "day5" / "day5_l4_benchmark_20261007-2337" / "scaling" / "scaling.json")
    args = parser.parse_args(argv)

    data = Inputs(args.run.resolve(), args.cpu.resolve(), args.scaling.resolve())
    rows = before_after_rows(data)
    outputs = {
        "before_vs_after.md": before_after_markdown(data, rows),
        "retrieval_report.md": retrieval_markdown(data),
        "generation_report.md": generation_markdown(data),
        "latency_report.md": latency_markdown(data),
        "concurrency_report.md": concurrency_markdown(data),
    }
    for name, text in outputs.items():
        (RESULTS / name).write_text(text, encoding="utf-8", newline="\n")
    (RESULTS / "summary.json").write_text(json.dumps(summary_json(data, rows), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    update_metrics_files(data)
    print("\n".join(f"wrote results/{name}" for name in [*outputs, "summary.json"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
