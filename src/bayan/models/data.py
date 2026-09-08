"""Lab 3 starter: dataset construction and split integrity."""
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict

DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "raw" / "bayan_feedback.csv"
SPLIT_NAMES = ("train", "validation", "test")


def build_topic_dataset(csv_path: Path = DATA_PATH, group_col: str = "citizen_group_id"):
    """Load bayan_feedback.csv and return a DatasetDict with grouped splits.

    A citizen (``citizen_group_id``) must never appear in more than one split,
    otherwise the classifier could memorise a citizen's writing style/topic
    instead of generalising. We verify that before trusting the supplied
    ``split`` column, rather than assuming it is already clean.
    """
    df = pd.read_csv(csv_path)

    group_to_splits = df.groupby(group_col)["split"].nunique()
    leaking_groups = group_to_splits[group_to_splits > 1]
    if not leaking_groups.empty:
        raise ValueError(
            f"{len(leaking_groups)} citizen_group_id value(s) leak across splits: "
            f"{leaking_groups.index.tolist()[:5]}"
        )

    splits = {}
    for name in SPLIT_NAMES:
        subset = df[df["split"] == name].reset_index(drop=True)
        splits[name] = Dataset.from_pandas(subset, preserve_index=False)

    return DatasetDict(splits)
