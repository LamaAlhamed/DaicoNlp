"""Lab 4: 30 supplied Arabic golden pairs."""
import pandas as pd
import pytest
from bayan.preprocessing.arabic import ArabicProfile, normalize_arabic

DF = pd.read_csv("data/eval/arabic_normalize_golden.csv")

@pytest.mark.parametrize("row", [r for _, r in DF.iterrows()], ids=DF["example_id"].tolist())
def test_arabic_golden(row):
    profile = ArabicProfile(name=row["profile"], dediacritize=True)
    assert normalize_arabic(row["raw_text"], profile) == row["expected_normalized"]
