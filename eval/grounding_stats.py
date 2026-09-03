"""Interval estimation for the grounding-gate evaluation.

Four estimators, each chosen for the shape of the question it answers:

- **Wilson score interval** for a single rate (false-flag rate, per-class
  detection rate). Wilson rather than the normal approximation because these
  rates sit near 0 or 1 at n in the low hundreds, exactly where the normal
  interval runs past the [0, 1] boundary and stops meaning anything.
- **Clopper-Pearson exact interval**, used specifically for the small-n
  measurement pre-registered in DECISIONS.md 2026-09-03 ("Grounding-gate
  measurement, small-n run pre-registered"). It inverts the exact binomial CDF
  rather than approximating it, so unlike Wilson it stays valid - not just
  close - at the very small numerators (including zero) this measurement can
  land on at n=45. It is conservative (wider) than Wilson at the same n; that
  is the correct trade at this n, not a defect.
- **Exact McNemar** for "does the gate beat the baseline on this class".
  Both arms make a hard binary decision on the *same* item, so the paired
  information is entirely in the discordant pairs, and the exact binomial test
  on those pairs is the standard estimator for it. `eval/extraction_comparison.py`
  records why McNemar was *not* used there - it would have needed a decision
  threshold that neither arm's AUC measurement had. Here both arms already
  make the decision, so the objection does not apply and the appropriate test
  is different. The lesson carried across is the pairing, not the estimator.
- **Paired bootstrap over items** for an interval on the *difference* in
  rates, since McNemar returns a p-value and not an interval. Every draw takes
  the same item indices from both arms, which is what preserves the pairing.

No LLM. No network.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import beta, binomtest

DEFAULT_CONFIDENCE = 0.95
DEFAULT_N_BOOTSTRAP = 10_000

_Z = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}


@dataclass(frozen=True)
class Rate:
    """A proportion that always travels with its n and its interval.

    `method` names which interval was used ("Wilson" or "Clopper-Pearson") -
    added because both `wilson()` and `clopper_pearson()` build this same
    class, and a `__str__` that always said "Wilson CI" would mislabel every
    Clopper-Pearson result it prints.
    """

    label: str
    numerator: int
    denominator: int
    ci_low: float
    ci_high: float
    confidence: float
    method: str = "Wilson"

    @property
    def value(self) -> float:
        return self.numerator / self.denominator if self.denominator else float("nan")

    def __str__(self) -> str:
        return (
            f"{self.label}: {self.value:.3f} ({self.numerator}/{self.denominator}, "
            f"{int(self.confidence * 100)}% {self.method} CI {self.ci_low:.3f}-{self.ci_high:.3f})"
        )


def wilson(
    numerator: int, denominator: int, label: str = "", confidence: float = DEFAULT_CONFIDENCE
) -> Rate:
    if denominator <= 0:
        return Rate(label, numerator, denominator, float("nan"), float("nan"), confidence, "Wilson")
    z = _Z[confidence]
    p = numerator / denominator
    denom = 1.0 + z * z / denominator
    center = (p + z * z / (2 * denominator)) / denom
    half = (
        z / denom * math.sqrt(p * (1 - p) / denominator + z * z / (4 * denominator * denominator))
    )
    return Rate(
        label,
        numerator,
        denominator,
        max(0.0, center - half),
        min(1.0, center + half),
        confidence,
        "Wilson",
    )


def clopper_pearson(
    numerator: int, denominator: int, label: str = "", confidence: float = DEFAULT_CONFIDENCE
) -> Rate:
    """Exact binomial CI: inverts the Beta CDF rather than approximating it,
    so it is valid (not just asymptotically close) at the small numerators -
    including zero - this measurement's n=45 can produce. Standard
    Beta-quantile construction: the lower bound comes from Beta(k, n-k+1) and
    the upper from Beta(k+1, n-k), each collapsing to the [0, 1] boundary at
    k=0 or k=n rather than requiring a special case.
    """
    if denominator <= 0:
        return Rate(
            label, numerator, denominator, float("nan"), float("nan"), confidence, "Clopper-Pearson"
        )
    alpha = 1.0 - confidence
    lo = 0.0 if numerator == 0 else beta.ppf(alpha / 2, numerator, denominator - numerator + 1)
    hi = (
        1.0
        if numerator == denominator
        else beta.ppf(1 - alpha / 2, numerator + 1, denominator - numerator)
    )
    return Rate(label, numerator, denominator, float(lo), float(hi), confidence, "Clopper-Pearson")


@dataclass(frozen=True)
class PairedComparison:
    """Gate versus baseline on identical items."""

    label: str
    n_items: int
    gate: Rate
    baseline: Rate
    both_correct: int
    gate_only: int  # gate correct, baseline wrong
    baseline_only: int  # baseline correct, gate wrong
    neither: int
    p_value: float
    difference: float
    ci_low: float
    ci_high: float
    confidence: float

    @property
    def excludes_zero(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def __str__(self) -> str:
        verdict = "excludes zero" if self.excludes_zero else "INCLUDES zero"
        return (
            f"{self.label} on n={self.n_items}: gate {self.gate.value:.3f} vs "
            f"baseline {self.baseline.value:.3f}, difference {self.difference:+.3f} "
            f"({int(self.confidence * 100)}% paired bootstrap CI {self.ci_low:+.3f} to "
            f"{self.ci_high:+.3f}, {verdict}); McNemar discordant "
            f"{self.gate_only}/{self.baseline_only}, exact p={self.p_value:.4f}"
        )


def _bootstrap_difference(
    gate_correct: np.ndarray,
    baseline_correct: np.ndarray,
    n_bootstrap: int,
    confidence: float,
    random_state: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(random_state)
    n = gate_correct.size
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    differences = gate_correct[idx].mean(axis=1) - baseline_correct[idx].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(differences, [tail, 1.0 - tail])
    return float(low), float(high)


def paired_comparison(
    label: str,
    gate_correct,
    baseline_correct,
    confidence: float = DEFAULT_CONFIDENCE,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    random_state: int = 0,
) -> PairedComparison:
    """`gate_correct[i]` and `baseline_correct[i]` must describe the same item.

    "Correct" is per-class: on a positive class it means the arm flagged the
    item; on the clean class the caller passes "did not flag", so a higher
    number is better in both directions and the difference reads the same way.
    """
    gate = np.asarray(gate_correct, dtype=bool)
    baseline = np.asarray(baseline_correct, dtype=bool)
    if gate.shape != baseline.shape:
        raise ValueError(f"paired arms must be the same length, got {gate.shape}, {baseline.shape}")
    if gate.size == 0:
        raise ValueError("cannot compare arms on an empty item set")

    gate_only = int(np.sum(gate & ~baseline))
    baseline_only = int(np.sum(~gate & baseline))
    discordant = gate_only + baseline_only
    p_value = float(binomtest(gate_only, discordant, 0.5).pvalue) if discordant else 1.0

    ci_low, ci_high = _bootstrap_difference(
        gate.astype(float), baseline.astype(float), n_bootstrap, confidence, random_state
    )
    return PairedComparison(
        label=label,
        n_items=int(gate.size),
        gate=wilson(int(gate.sum()), int(gate.size), f"{label} gate", confidence),
        baseline=wilson(int(baseline.sum()), int(baseline.size), f"{label} baseline", confidence),
        both_correct=int(np.sum(gate & baseline)),
        gate_only=gate_only,
        baseline_only=baseline_only,
        neither=int(np.sum(~gate & ~baseline)),
        p_value=p_value,
        difference=float(gate.mean() - baseline.mean()),
        ci_low=ci_low,
        ci_high=ci_high,
        confidence=confidence,
    )
