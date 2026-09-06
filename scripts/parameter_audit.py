"""Lab 2 starter: parameter accounting for mBERT and CAMeLBERT."""


def audit(checkpoint: str) -> dict:
    # TODO(Lab 2): bucket embeddings / attention / FFN / norms / pooler / other.
    raise NotImplementedError


if __name__ == "__main__":
    for ckpt in [
        "bert-base-multilingual-cased",
        "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    ]:
        print(ckpt, audit(ckpt))
