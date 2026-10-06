from __future__ import annotations

import argparse
import json
import random
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


class GroupAwareBatchSampler:
    def __init__(self, groups: list[int], batch_size: int, drop_last: bool, seed: int) -> None:
        self.groups = groups
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def plan(self, epoch: int) -> list[list[int]]:
        remaining = list(range(len(self.groups)))
        random.Random(self.seed + epoch).shuffle(remaining)
        batches = []
        while remaining:
            batch, used, deferred = [], set(), []
            for index in remaining:
                if len(batch) < self.batch_size and self.groups[index] not in used:
                    batch.append(index)
                    used.add(self.groups[index])
                else:
                    deferred.append(index)
            if self.drop_last and len(batch) < self.batch_size:
                break
            batches.append(batch)
            remaining = deferred
        return batches

    def __iter__(self):
        batches = self.plan(self.epoch)
        self.epoch += 1
        return iter(batches)

    def __len__(self) -> int:
        return len(self.plan(self.epoch))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune the embedding model on knowledge bank retrieval pairs.")
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--train-file", default="training/embedding_train.jsonl")
    parser.add_argument("--validation-file", default="training/ir_validation.json")
    parser.add_argument("--hub-dataset", default=None)
    parser.add_argument("--output-dir", default="artifacts/embedding")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args(argv)

    import torch
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss

    config = yaml.safe_load(resolve(args.config, args.hub_dataset).read_text(encoding="utf-8"))["embedding"]
    rows = read_jsonl(resolve(args.train_file, args.hub_dataset))
    validation = json.loads(resolve(args.validation_file, args.hub_dataset).read_text(encoding="utf-8"))
    query_prompt = f"Instruct: {config['query_instruction']}\nQuery:"

    groups = [row["group"] for row in rows]
    negative_columns = sorted(key for key in rows[0] if key.startswith("negative_"))
    columns = ["anchor", "positive", *negative_columns]
    train_dataset = Dataset.from_dict({column: [row[column] for row in rows] for column in columns})

    model = SentenceTransformer(config["base_model"], device="cuda" if torch.cuda.is_available() else "cpu")
    model.max_seq_length = config["max_seq_length"]
    evaluator = InformationRetrievalEvaluator(
        queries=validation["queries"],
        corpus=validation["corpus"],
        relevant_docs={query_id: set(chunk_ids) for query_id, chunk_ids in validation["relevant"].items()},
        name="val",
        query_prompt=query_prompt,
        accuracy_at_k=[1, 5, 10],
        precision_recall_at_k=[1, 5, 10, 20],
        mrr_at_k=[10],
        ndcg_at_k=[10],
        map_at_k=[10],
        batch_size=32,
        show_progress_bar=False,
    )
    baseline = evaluator(model)

    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=config["mini_batch_size"])
    output_dir = Path(args.output_dir)
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        bf16=config["bf16"] and torch.cuda.is_available(),
        prompts={"anchor": query_prompt},
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=config["selection_metric"],
        greater_is_better=True,
        logging_steps=5,
        seed=config["seed"],
        report_to="none",
    )

    class GroupAwareTrainer(SentenceTransformerTrainer):
        def get_batch_sampler(self, dataset, batch_size, drop_last, *unused_args, **unused_kwargs):
            return GroupAwareBatchSampler(groups, batch_size, drop_last, config["seed"])

    trainer = GroupAwareTrainer(
        model=model, args=training_args, train_dataset=train_dataset, loss=loss, evaluator=evaluator
    )
    started = time.time()
    train_output = trainer.train()
    training_seconds = time.time() - started
    final = evaluator(model)

    final_dir = output_dir / "final"
    model.save(str(final_dir))
    metrics = {
        "base_model": config["base_model"],
        "method": config["method"],
        "loss": config["loss"],
        "hyperparameters": {key: config[key] for key in ("batch_size", "mini_batch_size", "epochs", "learning_rate", "warmup_ratio", "weight_decay", "max_seq_length", "seed")},
        "training_rows": len(rows),
        "negatives_per_row": len(negative_columns),
        "training_seconds": round(training_seconds, 1),
        "global_steps": train_output.global_step,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "validation_before": baseline,
        "validation_after": final,
        "log_history": trainer.state.log_history,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    (final_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    summary = {key: (metrics["validation_before"].get(f"val_cosine_{key}"), metrics["validation_after"].get(f"val_cosine_{key}")) for key in ("ndcg@10", "mrr@10", "accuracy@1", "recall@5")}
    print(json.dumps({"validation_before_after": summary, "training_seconds": metrics["training_seconds"], "best_checkpoint": metrics["best_checkpoint"]}, indent=2))

    if args.push:
        model.push_to_hub(config["hub_model_id"], private=True, commit_message="Fine-tuned on Qimam knowledge bank retrieval pairs")
        from huggingface_hub import upload_file

        upload_file(
            path_or_fileobj=str(final_dir / "training_metrics.json"),
            path_in_repo="training_metrics.json",
            repo_id=config["hub_model_id"],
            commit_message="Add training metrics",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
