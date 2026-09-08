"""Lab 3A starter: fine-tune the Bayan topic classifier."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from bayan.models.data import build_topic_dataset

CHECKPOINT = "xlm-roberta-base"  # Lab 1 decision — see DECISIONS.md#tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="artifacts/topic_classifier",
        help="Where to save the trained classifier artefact (local path or mounted Drive path).",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = build_topic_dataset()

    labels = sorted(set(ds["train"]["topic"]))
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    def preprocess(batch):
        enc = tokenizer(batch["text"], truncation=True, max_length=128)
        enc["labels"] = [label2id[t] for t in batch["topic"]]
        return enc

    tokenized = ds.map(preprocess, batched=True)
    tokenized = tokenized.remove_columns(
        [c for c in ds["train"].column_names if c not in ("input_ids", "attention_mask", "labels")]
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    def compute_metrics(eval_pred):
        logits, refs = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"macro_f1": f1_score(refs, preds, average="macro")}

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    start = time.time()
    trainer.train()
    train_time = time.time() - start

    val_metrics = trainer.evaluate(tokenized["validation"])

    # Frozen test: evaluate ONCE, after all training/tuning decisions are locked in.
    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    with open(output_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": CHECKPOINT,
                "validation_macro_f1": val_metrics["eval_macro_f1"],
                "frozen_test_macro_f1": test_metrics["test_macro_f1"],
                "train_time_seconds": train_time,
            },
            f,
            indent=2,
        )

    print(f"Validation macro-F1: {val_metrics['eval_macro_f1']:.4f}")
    print(f"Frozen test macro-F1: {test_metrics['test_macro_f1']:.4f}")
    print(f"Train time: {train_time:.2f}s")
    print(f"Saved artefact to: {output_dir}")


if __name__ == "__main__":
    main()
