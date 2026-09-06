"""Lab 3B alignment unit tests: 8 contracts, including continuation pieces."""
import pytest
from bayan.models.ner import align_labels

CASES = [
    ([None, 0, None], [3], [-100, 3, -100]),
    ([None, 0, 0, None], [1], [-100, 1, -100, -100]),
    ([None, 0, 1, None], [1, 2], [-100, 1, 2, -100]),
    ([None, 0, 0, 1, None], [4, 0], [-100, 4, -100, 0, -100]),
    ([None, 0, 1, 1, None], [0, 5], [-100, 0, 5, -100, -100]),
    ([0], [2], [2]),
    ([0, 0], [2], [2, -100]),
    ([None, 0, 0, 1, 1, None], [7, 8], [-100, 7, -100, 8, -100, -100]),
]

@pytest.mark.parametrize("word_ids,labels,expected", CASES)
def test_align_labels(word_ids, labels, expected):
    assert align_labels(word_ids, labels) == expected
