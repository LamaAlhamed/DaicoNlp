"""Lab 1 compliance evidence: 100% recall on the provided 60-case fixture."""
import pandas as pd
from bayan.preprocessing.core import mask_pii


def test_pii_recall_is_100_percent():
    df = pd.read_csv("data/eval/pii_test_set.csv")
    actual = df["text"].map(mask_pii)
    expected = df["expected_masked"]

    hits = int((actual == expected).sum())
    total = len(df)
    recall = hits / total

    assert total == 60, f"Expected the provided 60-case fixture, found {total} rows"
    assert recall == 1.0, f"PII recall {recall:.1%} ({hits}/{total}); required 100%"
