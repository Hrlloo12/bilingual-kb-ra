from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml


def resolve(path: str, hub_dataset: str | None) -> Path:
    local = Path(path)
    if local.exists() or not hub_dataset:
        return local
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(hub_dataset, path, repo_type="dataset"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score_candidates(model, candidates: list[dict], passages: dict[str, str], batch_size: int) -> tuple[dict, float]:
    pairs = [(row["query"], passages[chunk_id]) for row in candidates for chunk_id in row["chunk_ids"]]
    started = time.time()
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
    elapsed = time.time() - started
    output, offset = {}, 0
    for row in candidates:
        values = scores[offset : offset + len(row["chunk_ids"])]
        offset += len(row["chunk_ids"])
        output[row["query_id"]] = {"split": row["split"], "scores": [[chunk_id, round(float(value), 6)] for chunk_id, value in zip(row["chunk_ids"], values)]}
    return output, elapsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score candidates with the base reranker, fine-tune it, and score again.")
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--train-file", default="training/reranker_train.jsonl")
    parser.add_argument("--candidates-file", default="training/reranker_candidates.jsonl")
    parser.add_argument("--passages-file", default="training/passages.json")
    parser.add_argument("--hub-dataset", default=None)
    parser.add_argument("--output-dir", default="artifacts/reranker")
    parser.add_argument("--score-batch-size", type=int, default=64)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args(argv)

    import torch
    from datasets import Dataset
    from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    config = yaml.safe_load(resolve(args.config, args.hub_dataset).read_text(encoding="utf-8"))["reranker"]
    rows = read_jsonl(resolve(args.train_file, args.hub_dataset))
    candidates = read_jsonl(resolve(args.candidates_file, args.hub_dataset))
    passages = json.loads(resolve(args.passages_file, args.hub_dataset).read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = CrossEncoder(config["base_model"], max_length=config["max_length"], device=device, model_kwargs={"torch_dtype": torch.float16} if device == "cuda" else {})
    base_scores, base_seconds = score_candidates(model, candidates, passages, args.score_batch_size)
    (output_dir / "scores_base.json").write_text(json.dumps(base_scores), encoding="utf-8")
    del model
    torch.cuda.empty_cache()

    negative_columns = sorted(key for key in rows[0] if key.startswith("negative_"))
    pairs = {"query": [], "passage": [], "label": []}
    for row in rows:
        for column, label in [("positive", 1.0), *((column, 0.0) for column in negative_columns)]:
            pairs["query"].append(row["query"])
            pairs["passage"].append(row[column])
            pairs["label"].append(label)
    train_dataset = Dataset.from_dict(pairs)

    model = CrossEncoder(config["base_model"], max_length=config["max_length"], device=device)
    loss = BinaryCrossEntropyLoss(model, pos_weight=torch.tensor(config["pos_weight"]))
    training_args = CrossEncoderTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        bf16=config["bf16"] and device == "cuda",
        save_strategy="no",
        logging_steps=10,
        seed=config["seed"],
        report_to="none",
    )
    trainer = CrossEncoderTrainer(model=model, args=training_args, train_dataset=train_dataset, loss=loss)
    started = time.time()
    train_output = trainer.train()
    training_seconds = time.time() - started

    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    if device == "cuda":
        model.model.half()
    tuned_scores, tuned_seconds = score_candidates(model, candidates, passages, args.score_batch_size)
    (output_dir / "scores_finetuned.json").write_text(json.dumps(tuned_scores), encoding="utf-8")

    candidate_pairs = sum(len(row["chunk_ids"]) for row in candidates)
    metrics = {
        "base_model": config["base_model"],
        "method": config["method"],
        "loss": config["loss"],
        "hyperparameters": {key: config[key] for key in ("batch_size", "epochs", "learning_rate", "warmup_ratio", "weight_decay", "max_length", "pos_weight", "seed")},
        "training_rows": len(rows),
        "training_pairs": len(train_dataset),
        "negatives_per_row": len(negative_columns),
        "training_seconds": round(training_seconds, 1),
        "global_steps": train_output.global_step,
        "log_history": trainer.state.log_history,
        "candidate_pairs_scored": candidate_pairs,
        "scoring_seconds": {"base": round(base_seconds, 1), "finetuned": round(tuned_seconds, 1)},
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
    }
    (final_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({key: metrics[key] for key in ("training_pairs", "global_steps", "training_seconds", "scoring_seconds", "device")}, indent=2))

    if args.push:
        from huggingface_hub import HfApi

        model.model.float()
        model.push_to_hub(config["hub_model_id"], private=True, exist_ok=True)
        api = HfApi()
        for name in ("training_metrics.json",):
            api.upload_file(path_or_fileobj=str(final_dir / name), path_in_repo=name, repo_id=config["hub_model_id"])
        for name in ("scores_base.json", "scores_finetuned.json"):
            api.upload_file(path_or_fileobj=str(output_dir / name), path_in_repo=f"eval/{name}", repo_id=config["hub_model_id"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
