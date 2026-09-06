"""Lab 6 bootstrap correctness: six compact contracts."""
import math
import pytest
from bayan.evaluation.bootstrap import bootstrap_ci, paired_bootstrap_diff


def test_bootstrap_returns_three_values():
    out = bootstrap_ci([0, 1, 1, 0, 1], n_boot=200, seed=1)
    assert len(out) == 3


def test_bootstrap_interval_is_ordered():
    point, lo, hi = bootstrap_ci([0, 1, 1, 0, 1], n_boot=200, seed=1)
    assert lo <= point <= hi


def test_bootstrap_is_reproducible_with_seed():
    a = bootstrap_ci([0, 1, 1, 0, 1], n_boot=200, seed=7)
    b = bootstrap_ci([0, 1, 1, 0, 1], n_boot=200, seed=7)
    assert a == b


def test_paired_diff_zero_for_identical_inputs():
    delta, lo, hi = paired_bootstrap_diff([1, 0, 1, 1], [1, 0, 1, 1], n_boot=200, seed=2)
    assert math.isclose(delta, 0.0)
    assert lo <= 0 <= hi


def test_paired_diff_preserves_pairing():
    delta, lo, hi = paired_bootstrap_diff([1, 1, 1, 1], [0, 0, 0, 0], n_boot=200, seed=2)
    assert delta > 0
    assert lo > 0


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1, 0], [1], n_boot=20, seed=1)
