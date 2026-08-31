"""Unit tests for the closed-form oracle sweep's arithmetic. These check that
`oracle_pr_auc` implements the step-function AP formula correctly - they do
NOT verify that the closed form estimates the right real-world quantity.
That verification (GENERATOR.md §5's own specified method: replicate label
draws, not a hand-picked idealization) lives in
tests/test_eval_oracle_replicate_check.py. A prior version of this file used
GENERATOR.md §5's rough two-point idealization (p=0.08/0.39, no overlap) as a
stand-in for that verification; that idealization was the same reasoning that
produced the 0.30-0.36 ceiling guess the real measurement later overturned
(DECISIONS.md's 2026-08-31 "Phase 2 model quality, measured" entry), so
agreement with it proves nothing about correctness - it is kept below only as
a hand-computable arithmetic check on the formula.
"""

import numpy as np
import pytest

from eval.oracle import oracle_pr_auc, oracle_precision_recall_curve, prevalence_baseline


def test_oracle_pr_auc_formula_matches_a_hand_computed_two_point_case():
    """Arithmetic check only: for two point-masses of p with no overlap, AP
    has a hand-computable closed form (worked in GENERATOR.md §5: AP ~= 0.37
    for p=0.08/0.39 at a 45/55 split). This confirms `oracle_pr_auc` computes
    that formula correctly - it is not evidence about the real dataset's
    ceiling, which is a different, overlapping, empirically measured
    distribution of p.
    """
    n = 100_000
    n_low = int(0.45 * n)
    n_high = n - n_low
    p = np.concatenate([np.full(n_low, 0.08), np.full(n_high, 0.39)])

    ap = oracle_pr_auc(p)

    assert ap == pytest.approx(0.37, abs=0.01)


def test_oracle_precision_at_the_lowest_threshold_equals_mean_p():
    p = np.array([0.1, 0.2, 0.3, 0.4])
    thresholds, precisions, recalls = oracle_precision_recall_curve(p)

    assert thresholds[0] == pytest.approx(p.min())
    assert precisions[0] == pytest.approx(p.mean())
    assert recalls[0] == pytest.approx(1.0)


def test_oracle_recall_at_the_highest_threshold_is_that_points_share_of_total_p():
    p = np.array([0.1, 0.2, 0.3, 0.4])
    thresholds, _precisions, recalls = oracle_precision_recall_curve(p)

    assert thresholds[-1] == pytest.approx(p.max())
    assert recalls[-1] == pytest.approx(p.max() / p.sum())


def test_oracle_pr_auc_is_at_least_prevalence_for_any_non_degenerate_p():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.75, size=5000)
    ap = oracle_pr_auc(p)
    assert ap >= p.mean() - 1e-9


def test_prevalence_baseline_is_the_mean_label():
    labels = np.array([True, True, False, False, False])
    assert prevalence_baseline(labels) == pytest.approx(0.4)
