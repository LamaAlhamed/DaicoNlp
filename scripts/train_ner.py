"""Lab 3B starter: fine-tune token classification with correct alignment."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict
from seqeval.metrics import classification_report, f1_score as seqeval_f1
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from bayan.models.ner import align_labels

CHECKPOINT = "xlm-roberta-base"  # Lab 1 decision — see DECISIONS.md#tokenizer
CONLL_PATH = Path("data/models/bayan_ner.conll")
SEED = 42


def read_conll(path: Path):
    """Parse a CoNLL file (word<TAB>tag per line, blank line = sentence break)."""
    sentences, all_tags = [], []
    words, tags = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if words:
                    sentences.append(words)
                    all_tags.append(tags)
                    words, tags = [], []
                continue
            word, tag = line.split("\t")
            words.append(word)
            tags.append(tag)
    if words:
        sentences.append(words)
        all_tags.append(tags)
    return sentences, all_tags


def split_sentences(sentences, tags, train_frac=0.7, val_frac=0.2, seed=SEED):
    """Deterministic shuffled 70/20/10 split (no grouping key applies to CoNLL data)."""
    indices = list(range(len(sentences)))
    random.Random(seed).shuffle(indices)
    n = len(indices)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    parts = {
        "train": indices[:n_train],
        "validation": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:],
    }
    return {
        name: {
            "tokens": [sentences[i] for i in idx],
            "ner_tags": [tags[i] for i in idx],
        }
        for name, idx in parts.items()
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="artifacts/ner",
        help="Where to save the trained NER artefact (local path or mounted Drive path).",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sentences, tags = read_conll(CONLL_PATH)
    split = split_sentences(sentences, tags)

    label_list = sorted({tag for seq in tags for tag in seq})
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}

    ds = DatasetDict({name: Dataset.from_dict(data) for name, data in split.items()})

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    def preprocess(batch):
        enc = tokenizer(batch["tokens"], truncation=True, is_split_into_words=True, max_length=256)
        all_labels = []
        for i, tag_seq in enumerate(batch["ner_tags"]):
            word_ids = enc.word_ids(batch_index=i)
            word_label_ids = [label2id[t] for t in tag_seq]
            all_labels.append(align_labels(word_ids, word_label_ids))
        enc["labels"] = all_labels
        return enc

    tokenized = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)

    model = AutoModelForTokenClassification.from_pretrained(
        CHECKPOINT, num_labels=len(label_list), id2label=id2label, label2id=label2id
    )
    collator = DataCollatorForTokenClassification(tokenizer)

    def compute_metrics(eval_pred):
        logits, refs = eval_pred
        preds = np.argmax(logits, axis=-1)
        true_labels, true_preds = [], []
        for pred_seq, ref_seq in zip(preds, refs):
            cur_labels, cur_preds = [], []
            for p, r in zip(pred_seq, ref_seq):
                if r == -100:
                    continue
                cur_labels.append(id2label[r])
                cur_preds.append(id2label[p])
            true_labels.append(cur_labels)
            true_preds.append(cur_preds)
        return {"entity_f1": seqeval_f1(true_labels, true_preds)}

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="entity_f1",
        logging_steps=50,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    start = time.time()
    trainer.train()
    train_time = time.time() - start

    val_metrics = trainer.evaluate(tokenized["validation"])

    # Frozen test: evaluate ONCE.
    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    with open(output_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": CHECKPOINT,
                "validation_entity_f1": val_metrics["eval_entity_f1"],
                "frozen_test_entity_f1": test_metrics["test_entity_f1"],
                "train_time_seconds": train_time,
            },
            f,
            indent=2,
        )

    print(f"Validation entity-F1: {val_metrics['eval_entity_f1']:.4f}")
    print(f"Frozen test entity-F1: {test_metrics['test_entity_f1']:.4f}")
    print(f"Train time: {train_time:.2f}s")
    print(f"Saved artefact to: {output_dir}")


if __name__ == "__main__":
    main()
