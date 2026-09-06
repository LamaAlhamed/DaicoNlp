"""Lab 3B QA post-processing contracts."""
import numpy as np
from bayan.models.qa import best_span


def test_qa_can_return_null():
    start = np.array([10.0, 1.0, 0.5])
    end = np.array([10.0, 1.0, 0.5])
    offsets = [None, (0, 4), (5, 9)]
    out = best_span(start, end, offsets, null_score=20.0, null_threshold=1.0)
    assert out.get("answer") is None


def test_qa_rejects_inverted_span():
    start = np.array([0.0, 1.0, 9.0])
    end = np.array([0.0, 8.0, 1.0])
    offsets = [None, (0, 4), (5, 9)]
    out = best_span(start, end, offsets, null_score=0.0, null_threshold=100.0)
    assert out.get("answer") is not None
