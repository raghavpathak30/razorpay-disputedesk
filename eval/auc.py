"""Batched ROC AUC.

Extracted from `eval.extraction_comparison` on 2026-09-02 (remediation item
2.1) so the leakage guard can use the same implementation the paired bootstrap
uses. Its correctness is load-bearing for every confidence interval in this
repository, and `tests/test_eval_auc_batch_property.py` fuzzes it against
`sklearn.metrics.roc_auc_score` over ~18,000 generated inputs at 1e-12
absolute tolerance rather than trusting the derivation.
"""

import numpy as np


def auc_batch(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """ROC AUC for a whole batch of (label, score) rows at once.

    `roc_auc_score` called 10,000 times per arm is the dominant cost of the
    bootstrap (~16s for n=60, minutes at n=200), which is enough to keep this
    out of CI. This is the standard rank formulation - AUC equals the
    Mann-Whitney U statistic normalised by `n_pos * n_neg` - computed over
    every row at once.

    Ties get **average** ranks, which is not an optional refinement here: a
    bootstrap draw samples with replacement, so duplicated items guarantee
    tied scores on every single draw. Ignoring ties would bias every
    bootstrap AUC. `tests/test_eval_extraction_comparison.py` pins this
    function against `sklearn.metrics.roc_auc_score` on tie-heavy input.

    Rows that are single-class return NaN, for the caller to drop.
    """
    n = y.shape[1]
    order = np.argsort(scores, axis=1, kind="stable")
    scores_sorted = np.take_along_axis(scores, order, axis=1)
    y_sorted = np.take_along_axis(y, order, axis=1)

    position = np.arange(1, n + 1, dtype=float)
    is_group_start = np.ones_like(scores_sorted, dtype=bool)
    is_group_start[:, 1:] = scores_sorted[:, 1:] != scores_sorted[:, :-1]
    is_group_end = np.ones_like(scores_sorted, dtype=bool)
    is_group_end[:, :-1] = scores_sorted[:, :-1] != scores_sorted[:, 1:]

    group_start = np.maximum.accumulate(np.where(is_group_start, position, 0.0), axis=1)
    group_end = np.minimum.accumulate(
        np.where(is_group_end, position, float(n + 1))[:, ::-1], axis=1
    )[:, ::-1]
    average_rank = (group_start + group_end) / 2.0

    n_pos = y_sorted.sum(axis=1).astype(float)
    n_neg = float(n) - n_pos
    sum_positive_ranks = (average_rank * y_sorted).sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (sum_positive_ranks - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return np.where((n_pos == 0) | (n_neg == 0), np.nan, auc)
