"""The paired-difference estimator every seed-paired comparison in `eval/`
reports through.

Why this module exists (2026-09-02 remediation, defect 1.1). Comparisons in
this project are paired by construction: a seed fixes the generated dataset,
the temporal split, and the trained model, and *both* arms are then scored on
that identical holdout. Seed-to-seed variation is large and it is shared - it
moves both arms together - which is exactly the variance a paired estimator
removes and an unpaired one leaves in.

The sweep summarised that design with `median(arm_a) - median(arm_b)`. Three
things are wrong with it:

- It is not an estimate of the paired difference. The two medians can come
  from different seeds, so the statistic answers a question nobody asked.
- It has no interval, so "is this difference real" had no way to be asked.
- It discards the direction of each individual pair, so a small difference of
  medians was read as "sign flips = noise" when the per-seed differences may
  be consistently one-signed.

The estimator here is the boring correct one: difference per pair, mean of
those differences, a percentile bootstrap interval resampling *pairs*, and
the sign-test count reported alongside so a reader can see consistency
separately from magnitude.
"""

from dataclasses import dataclass

import numpy as np

DEFAULT_N_BOOTSTRAP = 10_000
DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True)
class PairedDifference:
    """One paired comparison. `mean_difference` is the point estimate;
    `n_positive` out of `n_pairs` is the sign test, reported alongside rather
    than folded in, because "how large" and "how consistent" are different
    questions and a reader is entitled to both.
    """

    n_pairs: int
    mean_difference: float
    median_difference: float
    ci_low: float
    ci_high: float
    n_positive: int
    confidence: float

    @property
    def excludes_zero(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0


def paired_difference(
    arm_a,
    arm_b,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    confidence: float = DEFAULT_CONFIDENCE,
    random_state: int = 0,
) -> PairedDifference:
    """Paired mean difference of `arm_a - arm_b`, with a percentile bootstrap
    interval over pairs and a sign-test count.

    Element `i` of each array must be the same seed. Mismatched lengths raise
    rather than being zipped short: a silently truncated comparison would
    still produce a plausible-looking number.

    The bootstrap resamples *pair indices*, not values within an arm - that is
    what keeps the pairing intact under resampling. `random_state` fixes the
    resampling so a reported interval is reproducible; it is a reporting
    parameter, not a tuning one.
    """
    a = np.asarray(arm_a, dtype=float)
    b = np.asarray(arm_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired arms must be the same length, got {a.shape} and {b.shape}")
    if a.size == 0:
        raise ValueError("paired_difference needs at least one pair")

    differences = a - b
    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, differences.size, size=(n_bootstrap, differences.size))
    bootstrap_means = differences[indices].mean(axis=1)

    tail = (1.0 - confidence) / 2.0
    ci_low, ci_high = np.quantile(bootstrap_means, [tail, 1.0 - tail])

    return PairedDifference(
        n_pairs=int(differences.size),
        mean_difference=float(differences.mean()),
        median_difference=float(np.median(differences)),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n_positive=int((differences > 0).sum()),
        confidence=confidence,
    )


def format_paired(result: PairedDifference, unit: str = "") -> str:
    """One-line rendering for a README table cell or a CLI line. Always shows
    the interval and the sign count - a point estimate from this module should
    never appear on its own.
    """
    suffix = f" {unit}" if unit else ""
    return (
        f"{result.mean_difference:,.1f}{suffix} "
        f"({int(result.confidence * 100)}% CI {result.ci_low:,.1f} to {result.ci_high:,.1f}; "
        f"{result.n_positive}/{result.n_pairs} seeds positive)"
    )
