"""Paired comparison of two feature-extraction arms (remediation defect 1.2).

The README compared the LLM arm and the TF-IDF arm as two loose point
estimates - 0.4211 against 0.6371 - and concluded the gap was "wide enough
... to be a real result", "a wide margin, not a close call decided by noise".
Nothing in that comparison was paired, no interval was computed, and the two
numbers were not even known to have been measured on the same items or the
same number of items.

This module scores both arms on the **identical** items with the **identical**
cross-validation folds, and reports the AUC difference as a paired bootstrap
over items - so "how big" and "how sure" are separate, stated quantities.
"""

import numpy as np
import pytest

from eval.extraction_comparison import paired_auc_difference


def _labels_and_scores(n: int = 200):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=n).astype(bool)
    perfect = y.astype(float) + rng.normal(0, 0.01, size=n)
    noise = rng.random(n)
    return y, perfect, noise


def test_a_clearly_better_arm_produces_a_positive_difference_excluding_zero():
    y, perfect, noise = _labels_and_scores()

    result = paired_auc_difference(y, perfect, noise, random_state=0)

    assert result.auc_a > 0.95
    assert 0.3 < result.auc_b < 0.7
    assert result.difference > 0.25
    assert result.ci_low > 0.0
    assert result.excludes_zero is True


def test_identical_arms_have_exactly_zero_difference_and_an_interval_at_zero():
    """The control against an estimator that manufactures a gap."""
    y, perfect, _noise = _labels_and_scores()

    result = paired_auc_difference(y, perfect, perfect, random_state=0)

    assert result.difference == pytest.approx(0.0, abs=1e-12)
    assert result.ci_low == pytest.approx(0.0, abs=1e-9)
    assert result.ci_high == pytest.approx(0.0, abs=1e-9)
    assert result.excludes_zero is False


def test_two_equally_uninformative_arms_do_not_exclude_zero():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, size=200).astype(bool)
    a = rng.random(200)
    b = rng.random(200)

    result = paired_auc_difference(y, a, b, random_state=0)

    assert result.excludes_zero is False
    assert result.ci_low <= result.difference <= result.ci_high


def test_the_bootstrap_resamples_items_jointly_not_per_arm():
    """If each arm were resampled independently the pairing would be lost and
    the interval would be far too wide. Two arms that differ by a tiny
    constant shift must come back with a tight interval, not a wide one.
    """
    y, perfect, _noise = _labels_and_scores()
    nudged = perfect + 1e-6

    result = paired_auc_difference(y, perfect, nudged, random_state=0)

    assert abs(result.ci_high - result.ci_low) < 0.01


def test_the_bootstrap_is_reproducible_for_a_fixed_random_state():
    y, perfect, noise = _labels_and_scores()

    first = paired_auc_difference(y, perfect, noise, random_state=9)
    second = paired_auc_difference(y, perfect, noise, random_state=9)

    assert (first.ci_low, first.ci_high) == (second.ci_low, second.ci_high)


def test_arms_of_different_lengths_raise():
    y, perfect, noise = _labels_and_scores()

    with pytest.raises(ValueError):
        paired_auc_difference(y, perfect, noise[:-1], random_state=0)


def test_a_single_class_label_raises_rather_than_returning_a_nan():
    y = np.ones(50, dtype=bool)
    scores = np.linspace(0, 1, 50)

    with pytest.raises(ValueError):
        paired_auc_difference(y, scores, scores, random_state=0)


def test_n_items_is_reported_so_a_reader_can_judge_the_interval():
    y, perfect, noise = _labels_and_scores(n=137)

    result = paired_auc_difference(y, perfect, noise, random_state=0)

    assert result.n_items == 137


# --------------------------------------------------------------------------
# The hand-rolled batch AUC must agree with sklearn, ties included
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_distinct_scores", [2, 5, 40])
def test_batch_auc_matches_sklearn_including_on_tie_heavy_input(n_distinct_scores):
    """A bootstrap draw samples with replacement, so tied scores are present
    on every draw. This pins the average-rank tie handling against the
    reference implementation rather than trusting the derivation.
    """
    from sklearn.metrics import roc_auc_score

    from eval.extraction_comparison import _auc_batch

    rng = np.random.default_rng(4)
    rows = 25
    n = 40
    y = rng.integers(0, 2, size=(rows, n))
    y[:, 0] = 1  # guarantee both classes in every row
    y[:, 1] = 0
    scores = rng.integers(0, n_distinct_scores, size=(rows, n)).astype(float)

    batch = _auc_batch(y, scores)
    reference = np.array([roc_auc_score(y[i], scores[i]) for i in range(rows)])

    assert np.allclose(batch, reference)


def test_batch_auc_returns_nan_for_a_single_class_row():
    from eval.extraction_comparison import _auc_batch

    y = np.array([[1, 1, 1, 1]])
    scores = np.array([[0.1, 0.2, 0.3, 0.4]])

    assert np.isnan(_auc_batch(y, scores)).all()


# --------------------------------------------------------------------------
# "Does this arm beat chance" is a separate question from "does it beat that
# arm", and at small n the second can be unanswerable while the first is not
# --------------------------------------------------------------------------


def test_a_constant_score_vector_has_auc_exactly_one_half():
    """The construction `auc_vs_chance` relies on: all scores tied gives every
    item the same average rank, so AUC is exactly 0.5, not approximately.
    """
    from eval.extraction_comparison import _auc_batch

    y = np.array([[1, 0, 1, 0, 0, 1]])
    assert _auc_batch(y, np.zeros_like(y, dtype=float))[0] == pytest.approx(0.5)


def test_a_perfect_arm_is_distinguishable_from_chance():
    from eval.extraction_comparison import auc_vs_chance

    y, perfect, _noise = _labels_and_scores()

    result = auc_vs_chance(y, perfect, random_state=0)

    assert result.auc_b == pytest.approx(0.5)
    assert result.difference > 0.4
    assert result.excludes_zero is True


def test_a_pure_noise_arm_is_not_distinguishable_from_chance():
    from eval.extraction_comparison import auc_vs_chance

    y, _perfect, noise = _labels_and_scores()

    result = auc_vs_chance(y, noise, random_state=0)

    assert result.excludes_zero is False
