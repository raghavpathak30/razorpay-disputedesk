"""The paired estimator (remediation defect 1.1).

The cost sweep runs every seed through both arms - the policy and baseline A
see the identical generated dataset and the identical trained model. That is
a paired design, and it was summarised with a *difference of medians*:
`median(policy) - median(baseline_a)`. That statistic throws the pairing away.
It is not an estimate of the mean paired difference, it has no interval, and
it cannot say how many seeds actually moved in which direction - so a
per-point sign change was read as "noise" when the paired data may say
otherwise.

These tests pin the estimator itself against hand-computed values, including
a case constructed so the two estimators disagree by a wide margin.
"""

import numpy as np
import pytest

from eval.paired import paired_difference


def test_mean_of_differences_on_a_hand_computed_case():
    a = [10.0, 20.0, 30.0]
    b = [8.0, 17.0, 21.0]
    # differences: 2, 3, 9 -> mean 14/3
    result = paired_difference(a, b)

    assert result.n_pairs == 3
    assert result.mean_difference == pytest.approx(14 / 3)
    assert result.median_difference == pytest.approx(3.0)
    assert result.n_positive == 3


def test_the_difference_of_medians_and_the_paired_mean_can_disagree_wildly():
    """The defect, in miniature. Difference of medians says 1; the paired
    estimator - the one the design supports - says 33.
    """
    a = [1.0, 2.0, 100.0]
    b = [0.0, 3.0, 1.0]

    difference_of_medians = float(np.median(a) - np.median(b))
    result = paired_difference(a, b)

    assert difference_of_medians == pytest.approx(1.0)
    assert result.mean_difference == pytest.approx(33.0)
    assert result.n_positive == 2  # +1, -1, +99


def test_sign_count_counts_strictly_positive_differences_only():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [0.0, 2.0, 5.0, 1.0]  # differences: +1, 0, -2, +3
    result = paired_difference(a, b)

    assert result.n_positive == 2
    assert result.n_pairs == 4


def test_the_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    a = rng.normal(10.0, 1.0, size=40)
    b = rng.normal(9.0, 1.0, size=40)

    result = paired_difference(a, b, random_state=0)

    assert result.ci_low < result.mean_difference < result.ci_high


def test_a_zero_difference_interval_straddles_zero():
    """A genuinely null effect must not come back with an interval that
    excludes zero - the control that stops the estimator from manufacturing
    significance.
    """
    rng = np.random.default_rng(7)
    a = rng.normal(0.0, 1.0, size=200)
    b = a + rng.normal(0.0, 1.0, size=200) * 0.0  # identical pairs

    result = paired_difference(a, b, random_state=0)

    assert result.mean_difference == pytest.approx(0.0)
    assert result.ci_low <= 0.0 <= result.ci_high


def test_the_bootstrap_is_deterministic_for_a_fixed_random_state():
    rng = np.random.default_rng(3)
    a = rng.normal(5.0, 2.0, size=25)
    b = rng.normal(4.0, 2.0, size=25)

    first = paired_difference(a, b, random_state=11)
    second = paired_difference(a, b, random_state=11)

    assert first.ci_low == second.ci_low
    assert first.ci_high == second.ci_high


def test_mismatched_lengths_raise_rather_than_silently_truncating():
    with pytest.raises(ValueError):
        paired_difference([1.0, 2.0, 3.0], [1.0, 2.0])


def test_an_empty_input_raises():
    with pytest.raises(ValueError):
        paired_difference([], [])
