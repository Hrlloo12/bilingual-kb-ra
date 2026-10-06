from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

FIELDS = ("ar_msa", "ar_msa_long", "en", "en_long", "ar_dialect", "mixed")
JSON_SCHEMA = {
    "type": "object",
    "properties": {field: {"type": "string"} for field in FIELDS},
    "required": list(FIELDS),
    "additionalProperties": False,
}
SYSTEM_PROMPT = (
    "You write realistic search questions that customers, support agents and employees of Qimam Home, "
    "a Saudi home furniture retailer, type into the company knowledge base. You always answer with a single JSON object."
)
USER_TEMPLATE = """Write six different questions whose answer is the fact below.

Fact (English): {statement_en}
Fact (Arabic): {statement_ar}
Subject: {subject_en} / {subject_ar}

Rules:
- Every question must be fully answered by the fact and must clearly identify the subject.
- Never include the answer value itself in the question, and do not copy the fact sentence.
- ar_msa: Modern Standard Arabic, 6 to 20 words.
- ar_msa_long: Modern Standard Arabic, 25 to 50 words, a realistic situation followed by the question.
- en: natural English, 6 to 20 words.
- en_long: English, 25 to 50 words, a realistic situation followed by the question.
- ar_dialect: Saudi colloquial Arabic as people type it in chat (for example وش، أبغى، كم ياخذ، عشان، ليش), 5 to 20 words.
- mixed: an Arabic question that naturally code-switches with one to three English words such as price, warranty, policy, delivery, refund, leave, SLA, VPN, 5 to 20 words.
{name_rule}
Return JSON with the keys ar_msa, ar_msa_long, en, en_long, ar_dialect, mixed."""
NAME_RULES = {
    "latin": "- In the Arabic questions keep product or system names in Latin letters exactly as written in the fact.",
    "arabic_script": "- In ar_msa, ar_msa_long and ar_dialect write product or system names transliterated into Arabic script (for example Aria becomes آريا); in mixed keep them in Latin letters.",
}


def read_records(args: argparse.Namespace) -> list[dict]:
    if args.repo_id:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(args.repo_id, args.input_file, repo_type="dataset")
    else:
        path = args.input_file
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def build_messages(record: dict) -> list[dict]:
    prompt = USER_TEMPLATE.format(
        statement_en=record["statement_en"],
        statement_ar=record["statement_ar"],
        subject_en=record["subject_en"],
        subject_ar=record["subject_ar"],
        name_rule=NAME_RULES.get(record.get("name_style") or "", ""),
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]


def parse_output(text: str) -> dict | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate bilingual query paraphrases for knowledge bank facts with a local open model.")
    parser.add_argument("--repo-id", default=None, help="Hugging Face dataset repository holding the input and receiving the output.")
    parser.add_argument("--input-file", default="generation/fact_prompts.jsonl")
    parser.add_argument("--output-file", default="generation/raw_queries.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=900)
    args = parser.parse_args(argv)

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    records = read_records(args)
    llm = LLM(model=args.model, seed=args.seed, max_model_len=4096, gpu_memory_utilization=0.9)
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        guided_decoding=GuidedDecodingParams(json=JSON_SCHEMA),
    )
    outputs = llm.chat(
        [build_messages(record) for record in records],
        sampling,
        chat_template_kwargs={"enable_thinking": False},
    )
    results = []
    for record, output in zip(records, outputs, strict=True):
        text = output.outputs[0].text
        results.append(
            {
                "fact_id": record["fact_id"],
                "generator": args.model,
                "seed": args.seed,
                "temperature": args.temperature,
                "raw": text,
                "parsed": parse_output(text),
            }
        )
    payload = "\n".join(json.dumps(result, ensure_ascii=False) for result in results) + "\n"
    parsed_count = sum(1 for result in results if result["parsed"])
    print(json.dumps({"facts": len(records), "parsed": parsed_count}))

    if args.repo_id:
        from huggingface_hub import upload_file

        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory) / "raw_queries.jsonl"
            local.write_text(payload, encoding="utf-8")
            upload_file(
                path_or_fileobj=str(local),
                path_in_repo=args.output_file,
                repo_id=args.repo_id,
                repo_type="dataset",
                commit_message=f"Generated queries with {args.model} seed {args.seed}",
            )
    else:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_file).write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
