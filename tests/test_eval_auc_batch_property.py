"""Property tests for the hand-rolled batched AUC (remediation item 2.0).

`eval.auc.auc_batch` (written in Phase 1, extracted to its own module in 2.1)
was built to make the
bootstrap fast enough for CI. Every confidence interval in this repository now
runs through it, and the rebuilt leakage guard depends on it too, so one
pinned comparison against `sklearn.metrics.roc_auc_score` is not enough load
to put on it.

This file fuzzes it against sklearn over >1,000 randomly generated inputs
spanning the shapes that actually occur - bootstrap resamples are tie-heavy by
construction, and small n with a skewed base rate is the normal case here, not
an edge case - plus the degenerate inputs where a rank formulation is most
likely to be wrong.

**Tolerance: 1e-12 absolute.** Not an approximation tolerance - the two
computations should agree to floating-point noise, because both reduce to the
same rational number. Anything larger would be hiding a real disagreement. The
one place a looser bound would be defensible is very large n, where the
rank-sum accumulates more floating-point error than sklearn's trapezoidal
integration; n here stays well below that.

**Single-class behaviour: returns NaN, deliberately.** sklearn raises. NaN is
the right choice for a *batched* function whose whole purpose is bootstrap
resampling: a resample that happens to draw one class is an ordinary event,
not an error, and the caller (`paired_auc_difference`) drops those draws. A
raise would force per-row exception handling back into the hot loop, which is
what `auc_batch` exists to avoid. The behaviour is asserted below rather than
left to be discovered.
"""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from eval.auc import auc_batch

TOLERANCE = 1e-12
"""Absolute agreement required with `sklearn.metrics.roc_auc_score`."""


def _reference(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return np.array([roc_auc_score(y[i], scores[i]) for i in range(y.shape[0])])


def _force_both_classes(y: np.ndarray) -> np.ndarray:
    """sklearn refuses single-class rows, so the fuzz corpus guarantees both
    classes; the single-class case is covered separately below.
    """
    y = y.copy()
    y[:, 0] = 1
    y[:, 1] = 0
    return y


# --------------------------------------------------------------------------
# Fuzz: >1,000 generated inputs across n, base rate, and score granularity
# --------------------------------------------------------------------------

_FUZZ_CASES = [
    # (n, base rate, number of distinct score values, seed)
    (n, base_rate, n_distinct, seed)
    for n in (3, 5, 12, 37, 60, 200)
    for base_rate in (0.05, 0.2, 0.5, 0.8, 0.95)
    for n_distinct in (1, 2, 3, 10, 1000)
    for seed in range(4)
]


@pytest.mark.parametrize(("n", "base_rate", "n_distinct", "seed"), _FUZZ_CASES)
def test_batch_auc_agrees_with_sklearn(n, base_rate, n_distinct, seed):
    """`n_distinct=1` is the all-identical-scores case (every item tied);
    `n_distinct=2` and `3` are the heavy-tie cases a bootstrap resample
    produces; `1000` is effectively continuous.
    """
    rng = np.random.default_rng(seed * 1000 + n * 7 + n_distinct)
    rows = 30
    y = _force_both_classes((rng.random((rows, n)) < base_rate).astype(int))
    scores = rng.integers(0, n_distinct, size=(rows, n)).astype(float)

    assert np.allclose(auc_batch(y, scores), _reference(y, scores), atol=TOLERANCE, rtol=0)


def test_the_fuzz_corpus_is_at_least_a_thousand_inputs():
    """The brief asked for >=1,000 generated inputs; each parametrised case
    above scores 30 rows, so this is the count of individual AUC comparisons.
    """
    assert len(_FUZZ_CASES) * 30 >= 1000


@pytest.mark.parametrize("seed", range(12))
def test_batch_auc_agrees_with_sklearn_on_integer_scores(seed):
    """Integer scores are the case where average-rank tie handling matters
    most and where an off-by-one in the rank formula would show up.
    """
    rng = np.random.default_rng(seed)
    rows, n = 40, 25
    y = _force_both_classes(rng.integers(0, 2, size=(rows, n)))
    scores = rng.integers(-5, 6, size=(rows, n)).astype(float)

    assert np.allclose(auc_batch(y, scores), _reference(y, scores), atol=TOLERANCE, rtol=0)


@pytest.mark.parametrize("seed", range(12))
def test_batch_auc_agrees_with_sklearn_on_bootstrap_shaped_input(seed):
    """The input this function actually sees: a resample-with-replacement of a
    60-item set, so duplicated items and therefore duplicated scores.
    """
    rng = np.random.default_rng(seed)
    n = 60
    base_y = _force_both_classes(rng.integers(0, 2, size=(1, n)))[0]
    base_scores = rng.random(n)

    idx = rng.integers(0, n, size=(50, n))
    y = base_y[idx]
    scores = base_scores[idx]
    keep = np.array([len(np.unique(row)) > 1 for row in y])

    assert np.allclose(
        auc_batch(y[keep], scores[keep]), _reference(y[keep], scores[keep]),
        atol=TOLERANCE, rtol=0,
    )


# --------------------------------------------------------------------------
# Explicit edge cases
# --------------------------------------------------------------------------


def test_all_identical_scores_give_exactly_one_half():
    """Every item tied means no ranking information at all. Exactly 0.5, not
    approximately - `auc_vs_chance` depends on this being exact.
    """
    y = np.array([[1, 0, 1, 0, 0], [0, 1, 1, 1, 0]])
    scores = np.full_like(y, 7.0, dtype=float)

    assert np.allclose(auc_batch(y, scores), 0.5, atol=TOLERANCE, rtol=0)


def test_n_equals_two_with_one_of_each_class():
    """The smallest input for which AUC is defined: perfectly ordered is 1.0,
    inverted is 0.0.
    """
    y = np.array([[1, 0], [1, 0]])
    scores = np.array([[1.0, 0.0], [0.0, 1.0]])

    assert np.allclose(auc_batch(y, scores), [1.0, 0.0], atol=TOLERANCE, rtol=0)


def test_n_equals_two_tied():
    y = np.array([[1, 0]])
    scores = np.array([[0.5, 0.5]])

    assert np.allclose(auc_batch(y, scores), 0.5, atol=TOLERANCE, rtol=0)


@pytest.mark.parametrize("label", [0, 1])
def test_a_single_class_row_returns_nan_not_a_number(label):
    """Documented behaviour, deliberately different from sklearn (which
    raises). See this module's docstring for why NaN is right for a batched
    bootstrap primitive.
    """
    y = np.full((1, 6), label)
    scores = np.linspace(0.0, 1.0, 6).reshape(1, 6)

    result = auc_batch(y, scores)

    assert result.shape == (1,)
    assert np.isnan(result).all()


def test_n_equals_one_is_single_class_and_returns_nan():
    """AUC is undefined for one item - it cannot contain both classes."""
    assert np.isnan(auc_batch(np.array([[1]]), np.array([[0.4]]))).all()


def test_mixed_batch_returns_nan_only_for_the_degenerate_rows():
    """The property `paired_auc_difference` relies on: one bad resample must
    not poison the rest of the chunk.
    """
    y = np.array([[1, 0, 1, 0], [1, 1, 1, 1], [0, 1, 0, 1]])
    scores = np.array([[0.9, 0.1, 0.8, 0.2], [0.1, 0.2, 0.3, 0.4], [0.1, 0.9, 0.2, 0.8]])

    result = auc_batch(y, scores)

    assert not np.isnan(result[0])
    assert np.isnan(result[1])
    assert not np.isnan(result[2])
    assert np.allclose(result[[0, 2]], [1.0, 1.0], atol=TOLERANCE, rtol=0)


def test_exact_ties_straddling_the_decision_boundary():
    """Positives and negatives sharing the same score: the tied block must
    contribute exactly one half of its pairs, which is where a naive
    strict-inequality rank count would be wrong.
    """
    y = np.array([[1, 1, 0, 0]])
    scores = np.array([[0.5, 0.5, 0.5, 0.5]])
    assert np.allclose(auc_batch(y, scores), 0.5, atol=TOLERANCE, rtol=0)

    # One positive clearly above the tied block, one inside it.
    y2 = np.array([[1, 1, 0, 0]])
    scores2 = np.array([[1.0, 0.5, 0.5, 0.5]])
    assert np.allclose(
        auc_batch(y2, scores2), _reference(y2, scores2), atol=TOLERANCE, rtol=0
    )


def test_a_reversed_perfect_ranking_is_exactly_zero():
    y = np.array([[1, 1, 0, 0]])
    scores = np.array([[0.1, 0.2, 0.8, 0.9]])

    assert np.allclose(auc_batch(y, scores), 0.0, atol=TOLERANCE, rtol=0)


def test_boolean_labels_are_accepted_like_integer_ones():
    """Callers pass both - `auc_vs_chance` receives a bool array from the
    generator, the bootstrap passes ints.
    """
    y_bool = np.array([[True, False, True, False]])
    y_int = y_bool.astype(int)
    scores = np.array([[0.9, 0.1, 0.8, 0.2]])

    assert np.allclose(
        auc_batch(y_bool, scores), auc_batch(y_int, scores), atol=TOLERANCE, rtol=0
    )
