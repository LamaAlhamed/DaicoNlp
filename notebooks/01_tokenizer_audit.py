from pathlib import Path
import csv
import numpy as np
from transformers import AutoTokenizer

CANDIDATES = {
    "bert-base-multilingual-cased": "mBERT",
    "xlm-roberta-base": "XLM-R",
    "CAMeL-Lab/bert-base-arabic-camelbert-mix": "CAMeLBERT",
    "distilbert-base-uncased": "DistilBERT",
}

DATA = Path("data/raw/bayan_feedback.csv")


def fertility(tokenizer, texts) -> float:
    total_subwords = 0
    total_words = 0
    for text in texts:
        total_subwords += len(tokenizer.tokenize(text))
        total_words += len(text.split())
    return total_subwords / total_words


def load_texts_by_lang():
    ar_texts, en_texts = [], []
    with DATA.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["lang"] == "ar":
                ar_texts.append(row["text"])
            elif row["lang"] == "en":
                en_texts.append(row["text"])
    return ar_texts, en_texts


def main():
    ar_texts, en_texts = load_texts_by_lang()

    print(f"{'Tokenizer':<12} {'AR fertility':>13} {'EN fertility':>13} {'AR p95 len':>11} {'EN p95 len':>11}")
    for checkpoint, name in CANDIDATES.items():
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        ar_fert = fertility(tokenizer, ar_texts)
        en_fert = fertility(tokenizer, en_texts)

        ar_lengths = [len(tokenizer.encode(t)) for t in ar_texts]
        en_lengths = [len(tokenizer.encode(t)) for t in en_texts]

        ar_p95 = np.percentile(ar_lengths, 95)
        en_p95 = np.percentile(en_lengths, 95)

        print(f"{name:<12} {ar_fert:>13.2f} {en_fert:>13.2f} {ar_p95:>11.1f} {en_p95:>11.1f}")


if __name__ == "__main__":
    main()
