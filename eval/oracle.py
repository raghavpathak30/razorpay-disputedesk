"""The oracle Bayes ceiling (GENERATOR.md §5): the best any model could ever do
on this dataset, computed from the generator's true per-record `p` (a debug
column - never a model input, only used here for evaluation).

Closed form, not a two-point approximation: because each label is an
independent `Bernoulli(p_i)` draw, the *expected* precision and recall at any
threshold `t` follow directly from the distribution of `p` in the dataset:

    precision*(t) = mean(p_i for all i with p_i >= t)
    recall*(t)    = sum(p_i for all i with p_i >= t) / sum(p_i)

Sweeping `t` over the observed values of `p` traces the oracle precision-recall
curve; its area is the oracle PR-AUC.
"""

import numpy as np


def oracle_precision_recall_curve(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep every observed `p_i` as a threshold. Returns (thresholds, precision,
    recall), thresholds ascending.
    """
    p = np.asarray(p, dtype=float)
    thresholds = np.unique(p)
    total_p = p.sum()

    precisions = np.empty_like(thresholds)
    recalls = np.empty_like(thresholds)
    for i, t in enumerate(thresholds):
        selected = p[p >= t]
        precisions[i] = selected.mean()
        recalls[i] = selected.sum() / total_p

    return thresholds, precisions, recalls


def oracle_pr_auc(p: np.ndarray) -> float:
    """Average precision over the oracle precision-recall curve: the standard
    step-function AP, `sum((R_i - R_{i-1}) * P_i)` over recall points ordered
    ascending (R_0 = 0) - not a trapezoid between points, since precision is a
    step function of the threshold, not something to linearly interpolate
    between thresholds.

    This is a closed-form *expectation* over repeated `Bernoulli(p)` label
    draws, not a statistic of any single realized draw - it will not in
    general match `sklearn.metrics.average_precision_score(y_true, p)` for
    one particular `y_true` sample, only the mean of that quantity over many
    such samples. See `tests/test_eval_oracle_replicate_check.py`, which
    implements GENERATOR.md §5's own specified verification (replicate label
    draws, not the file's rough two-point illustration) and checks exactly
    that agreement.
    """
    _, precisions, recalls = oracle_precision_recall_curve(p)
    order = np.argsort(recalls)
    recalls_sorted = recalls[order]
    precisions_sorted = precisions[order]

    recall_deltas = np.diff(recalls_sorted, prepend=0.0)
    return float(np.sum(recall_deltas * precisions_sorted))


def prevalence_baseline(labels: np.ndarray) -> float:
    """A random (no-skill) classifier's PR-AUC equals the positive-class
    prevalence in the evaluated set - the number every PR-AUC in this project
    is reported alongside (GENERATOR.md §5, PHASES.md Phase 2).
    """
    return float(np.asarray(labels, dtype=float).mean())
