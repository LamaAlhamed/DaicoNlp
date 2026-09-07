from transformers import AutoModel


def audit(checkpoint: str) -> dict:
    model = AutoModel.from_pretrained(checkpoint)
    buckets = {"embeddings": 0, "attention": 0, "ffn": 0, "norms": 0, "pooler": 0, "other": 0}

    for name, param in model.named_parameters():
        n = param.numel()
        if "embeddings" in name:
            buckets["embeddings"] += n
        elif "pooler" in name:
            buckets["pooler"] += n
        elif "LayerNorm" in name:
            buckets["norms"] += n
        elif "attention" in name:
            buckets["attention"] += n
        elif "intermediate" in name or "output.dense" in name:
            buckets["ffn"] += n
        else:
            buckets["other"] += n

    total = sum(buckets.values())
    result = {"total": total}
    for k, v in buckets.items():
        result[k] = v
        result[f"{k}_pct"] = round(100 * v / total, 2)
    return result


if __name__ == "__main__":
    for ckpt in [
        "bert-base-multilingual-cased",
        "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    ]:
        print(ckpt)
        result = audit(ckpt)
        for k, v in result.items():
            print(f"  {k}: {v}")
        print()