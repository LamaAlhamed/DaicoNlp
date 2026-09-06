"""Lab 1 starter: audit four tokenizer candidates on Bayan AR/EN text."""
from pathlib import Path

CANDIDATES = {
    "bert-base-multilingual-cased": "mBERT",
    "xlm-roberta-base": "XLM-R",
    "CAMeL-Lab/bert-base-arabic-camelbert-mix": "CAMeLBERT",
    "distilbert-base-uncased": "DistilBERT",
}

DATA = Path("data/raw/bayan_feedback.csv")


def fertility(tokenizer, texts) -> float:
    # TODO(Lab 1): total subword pieces / whitespace words.
    raise NotImplementedError


def main():
    # TODO(Lab 1): load AR/EN slices, audit fertility and sequence lengths,
    # print a table, report p95 per language/tokenizer, and update BENCHMARKS.md.
    raise NotImplementedError("Complete the tokenizer audit in Lab 1")


if __name__ == "__main__":
    main()
