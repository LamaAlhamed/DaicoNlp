"""Lab 3B starter: run the 12-question QA smoke set."""
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from bayan.models.qa import best_span

# Public XLM-R checkpoint already fine-tuned on SQuAD 2.0 (same tokenizer
# family as the Lab 1 decision). Lab 3B does not include a separate
# "train a QA model" step, so we evaluate best_span() against a ready-made
# extractive-QA checkpoint rather than fine-tuning our own.
QA_CHECKPOINT = "deepset/xlm-roberta-base-squad2"

FULL_QA_PATH = Path("data/models/bayan_qa.json")
PROVIDED_SMOKE_PATH = Path("data/eval/qa_smoke_set.json")
SEED = 42
N_ANSWERABLE = 9
N_IMPOSSIBLE = 3


def load_squad_style(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    examples = []
    for article in data["data"]:
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                examples.append(
                    {
                        "id": qa["id"],
                        "question": qa["question"],
                        "context": context,
                        "is_impossible": qa["is_impossible"],
                        "gold_answer": qa["answers"][0]["text"] if qa["answers"] else None,
                    }
                )
    return examples


def build_smoke_set():
    """Build a 9-answerable + 3-unanswerable smoke set.

    NOTE / data gap: the supplied ``qa_smoke_set.json`` only contains
    answerable questions (0 impossible), so on its own it cannot exercise
    the null-handling contract the lab requires. We deduplicate its
    (repeated) answerable questions down to 9, then draw 3 impossible
    questions from the full ``bayan_qa.json`` pool with a fixed seed, so
    the smoke set is reproducible and actually covers the null path.
    """
    provided = load_squad_style(PROVIDED_SMOKE_PATH)

    seen_questions = set()
    unique_answerable = []
    for ex in provided:
        if not ex["is_impossible"] and ex["question"] not in seen_questions:
            unique_answerable.append(ex)
            seen_questions.add(ex["question"])

    full_pool = load_squad_style(FULL_QA_PATH)
    impossible_pool = [ex for ex in full_pool if ex["is_impossible"]]
    answerable_pool = [ex for ex in full_pool if not ex["is_impossible"]]

    rng = random.Random(SEED)

    answerable = list(unique_answerable)
    if len(answerable) < N_ANSWERABLE:
        needed = N_ANSWERABLE - len(answerable)
        answerable += rng.sample(answerable_pool, needed)
    answerable = answerable[:N_ANSWERABLE]

    impossible = rng.sample(impossible_pool, N_IMPOSSIBLE)

    return answerable + impossible


def run_example(model, tokenizer, example, device):
    encoding = tokenizer(
        example["question"],
        example["context"],
        truncation="only_second",
        max_length=384,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = encoding["offset_mapping"][0].tolist()
    sequence_ids = encoding.sequence_ids(0)
    # Only positions that belong to the context (sequence id 1) get real
    # offsets; question tokens / specials are masked to None for best_span().
    clean_offsets = [
        tuple(o) if sequence_ids[i] == 1 else None for i, o in enumerate(offsets)
    ]

    model_inputs = {k: v.to(device) for k, v in encoding.items() if k != "offset_mapping"}
    with torch.no_grad():
        outputs = model(**model_inputs)
    start_logits = outputs.start_logits[0].cpu().numpy()
    end_logits = outputs.end_logits[0].cpu().numpy()

    cls_index = 0  # [CLS]/<s> is always the first token for this tokenizer.
    null_score = float(start_logits[cls_index] + end_logits[cls_index])

    result = best_span(start_logits, end_logits, clean_offsets, null_score=null_score, null_threshold=0.0)

    if result["answer"] is None:
        return None
    start_char, end_char = result["answer"]
    return example["context"][start_char:end_char]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(QA_CHECKPOINT)
    model = AutoModelForQuestionAnswering.from_pretrained(QA_CHECKPOINT).to(device)
    model.eval()

    smoke_set = build_smoke_set()

    correct_answerable = 0
    correct_null = 0
    n_answerable = 0
    n_impossible = 0

    for ex in smoke_set:
        predicted = run_example(model, tokenizer, ex, device)
        if ex["is_impossible"]:
            n_impossible += 1
            hit = predicted is None
            correct_null += hit
            print(f"[NULL EXPECTED]  Q: {ex['question']!r}  predicted={predicted!r}  correct={hit}")
        else:
            n_answerable += 1
            hit = predicted is not None and predicted.strip() == (ex["gold_answer"] or "").strip()
            correct_answerable += hit
            print(
                f"[ANSWERABLE]     Q: {ex['question']!r}  gold={ex['gold_answer']!r}  "
                f"predicted={predicted!r}  correct={hit}"
            )

    print()
    print(f"Answerable spans correct: {correct_answerable}/{n_answerable}")
    print(f"Nulls correct: {correct_null}/{n_impossible}")


if __name__ == "__main__":
    main()
