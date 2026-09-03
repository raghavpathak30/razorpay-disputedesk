"""Compares two feature-extraction arms on identical items, paired.

Why this module exists (2026-09-02 remediation, defect 1.2). The LLM arm and
the TF-IDF arm were compared as two loose point estimates - 0.4211 against
0.6371 - and the gap was described in the README as "wide enough ... to be a
real result" and "a wide margin, not a close call decided by noise". That
sentence asserts a statistical property (not-noise) that nothing in the
comparison could support:

- The two numbers had no interval.
- They were not known to have been computed on the same items, or on the same
  number of items - the TF-IDF figure had no recorded n at all.
- Nothing was paired, so all the item-to-item variance the two arms share was
  left in.

What this module does instead: both arms produce an out-of-fold predicted
probability for **every one of the same items**, using the **same
cross-validation folds**, and the AUC difference is bootstrapped by resampling
*items* - the same item index taken from both arms on every draw, which is
what preserves the pairing.

An alternative the remediation brief allowed is McNemar on per-item
correctness. The paired AUC bootstrap is used instead because AUC is the
metric already recorded for both arms, and McNemar would require choosing a
decision threshold that neither arm's recorded measurement had.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from eval.auc import auc_batch

DEFAULT_N_BOOTSTRAP = 10_000
DEFAULT_CONFIDENCE = 0.95
_BOOTSTRAP_CHUNK = 2_000


def _chunk_sizes(total: int, chunk: int = _BOOTSTRAP_CHUNK) -> list[int]:
    sizes = [chunk] * (total // chunk)
    if total % chunk:
        sizes.append(total % chunk)
    return sizes


@dataclass(frozen=True)
class PairedAucDifference:
    """`difference` is `auc_a - auc_b` on the full item set; the interval is a
    percentile bootstrap over items. `n_items` is carried because an interval
    is uninterpretable without it - the original claim's central failure was
    that one arm's n was unknown.
    """

    n_items: int
    auc_a: float
    auc_b: float
    difference: float
    ci_low: float
    ci_high: float
    confidence: float

    @property
    def excludes_zero(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0


def out_of_fold_probabilities(
    X: np.ndarray, y: np.ndarray, n_splits: int = 5, random_state: int = 0
) -> np.ndarray:
    """Out-of-fold logistic-regression probabilities for an already-extracted
    numeric feature matrix - the LLM arm's typed fields.

    The TF-IDF arm has its own version
    (`eval.tfidf_baseline.tfidf_out_of_fold_probabilities`) because its
    vectorizer must be re-fit inside each training fold; both use the same
    `StratifiedKFold(shuffle=True, random_state=...)`, so with the same
    `random_state` the two arms are split identically and the comparison is
    genuinely paired.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        raise ValueError("cross-validated AUC needs both classes present in the labels")

    oof = np.empty(len(y), dtype=float)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, test_idx in cv.split(X, y):
        model = LogisticRegression(max_iter=1000, random_state=random_state)
        model.fit(X[train_idx], y[train_idx])
        oof[test_idx] = model.predict_proba(X[test_idx])[:, 1]
    return oof


def paired_auc_difference(
    true_label,
    scores_a,
    scores_b,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    confidence: float = DEFAULT_CONFIDENCE,
    random_state: int = 0,
) -> PairedAucDifference:
    """AUC(`scores_a`) - AUC(`scores_b`) against `true_label`, with a
    percentile bootstrap interval over items.

    Every bootstrap draw takes the *same* item indices from both arms. Drawing
    independently per arm would discard the pairing and inflate the interval,
    which is the failure `tests/test_eval_extraction_comparison.py`'s
    tiny-shift test guards against.

    Resampled draws that end up single-class carry no AUC and are skipped
    rather than counted as zero - counting them would pull the interval toward
    a difference of zero and understate a real gap.
    """
    y = np.asarray(true_label).astype(int)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if not (y.shape == a.shape == b.shape):
        raise ValueError(
            f"paired arms must be the same length, got {y.shape}, {a.shape}, {b.shape}"
        )
    if len(np.unique(y)) < 2:
        raise ValueError("AUC needs both classes present in the labels")

    auc_a = float(roc_auc_score(y, a))
    auc_b = float(roc_auc_score(y, b))

    rng = np.random.default_rng(random_state)
    differences = []
    # Chunked so a large `n_bootstrap` x large `n_items` batch cannot blow up
    # memory; the draws themselves are identical either way.
    for chunk in _chunk_sizes(n_bootstrap):
        idx = rng.integers(0, y.size, size=(chunk, y.size))
        chunk_differences = auc_batch(y[idx], a[idx]) - auc_batch(y[idx], b[idx])
        differences.append(chunk_differences[~np.isnan(chunk_differences)])
    all_differences = np.concatenate(differences)

    tail = (1.0 - confidence) / 2.0
    ci_low, ci_high = np.quantile(all_differences, [tail, 1.0 - tail])

    return PairedAucDifference(
        n_items=int(y.size),
        auc_a=auc_a,
        auc_b=auc_b,
        difference=auc_a - auc_b,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        confidence=confidence,
    )


def auc_vs_chance(
    true_label,
    scores,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    confidence: float = DEFAULT_CONFIDENCE,
    random_state: int = 0,
) -> PairedAucDifference:
    """One arm's AUC against chance (0.5), with the same bootstrap machinery.

    Asked separately from `paired_auc_difference` because "arm A beats arm B"
    and "arm B carries any signal at all" are different claims, and at a small
    n the second can be answerable when the first is not.

    Implemented by comparing against an all-constant score vector, whose AUC
    is exactly 0.5 by construction (every score tied gives every item the same
    average rank), so `difference` reads directly as "AUC above chance" and
    `excludes_zero` as "distinguishable from chance".
    """
    scores = np.asarray(scores, dtype=float)
    return paired_auc_difference(
        true_label,
        scores,
        np.zeros_like(scores),
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        random_state=random_state,
    )


def format_comparison(result: PairedAucDifference, name_a: str, name_b: str) -> str:
    """One line for the CLI and the README. Always carries n and the interval:
    a point estimate from this module must never travel alone.
    """
    verdict = "excludes zero" if result.excludes_zero else "INCLUDES zero"
    return (
        f"{name_a} AUC {result.auc_a:.4f} vs {name_b} AUC {result.auc_b:.4f} "
        f"on the same n={result.n_items} items: difference {result.difference:+.4f} "
        f"({int(result.confidence * 100)}% paired bootstrap CI "
        f"{result.ci_low:+.4f} to {result.ci_high:+.4f}, {verdict})"
    )
