"""Lab 3A grouped-split integrity contract."""
from bayan.models.data import build_topic_dataset


def test_grouped_split_has_no_citizen_overlap():
    ds = build_topic_dataset()
    train_ids = set(ds["train"]["citizen_group_id"])
    valid_ids = set(ds["validation"]["citizen_group_id"])
    test_ids = set(ds["test"]["citizen_group_id"])
    assert train_ids.isdisjoint(valid_ids)
    assert train_ids.isdisjoint(test_ids)
    assert valid_ids.isdisjoint(test_ids)
