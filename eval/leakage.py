"""The leakage guard: three independent checks that no feature column is
derivable from the label or from the generator's latents.

Why this module exists (2026-09-02 remediation, defect 2.1). The previous
guard was a single Pearson correlation threshold, `|r| > 0.9`, applied to
numeric columns only. It passed a frame containing an exact copy of the
generator's `p` and a perfect string leak simultaneously. Two separate reasons
it could not work:

1. **Correlation against a binary label has a ceiling set by the label's own
   noise, not by the leak.** `won_if_contested` is `Bernoulli(p)`. A verbatim
   copy of `p` — the most total leak possible, the model reading the answer
   sheet — correlates only about 0.36 with a draw from it. No threshold that
   also admits legitimate features could ever fire on that. The threshold was
   0.9.
2. **It skipped every non-numeric column**, so a column of literal
   `"recovered"`/`"written off"` strings was not examined at all.

The replacement is three guards that fail for different reasons, so a leak has
to defeat all three:

**(a) Provenance.** Set equality between the feature frame's columns and a
frozen allowlist, plus a value-hash of every feature column compared against
the value-hash of every latent column. No statistics involved, so this holds
however noisy the label is — it is the only one of the three that catches a
copied latent carrying no label signal at all (the generator's `epsilon`, say).

**(b) Discrimination ceiling.** The Bayes AUC is what ranking on the true `p`
achieves against the sampled label — the best any function of the causal
latents could do. Each feature's univariate AUC is compared against it. A copy
of `p`, or any monotone transform of one, sits exactly at the ceiling; a
legitimate noisy reading of a latent cannot. This is the right yardstick
because it is denominated in the same label noise the features are.

**(c) Categorical.** Normalised mutual information with the label, plus a
per-level purity check (any level with at least `min_level_rows` rows whose
label rate is exactly 0 or 1). Mutual information handles the many-level case
where a single tainted level would be diluted; the purity check handles the
case where it would not.
"""

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

from disputedesk.generator.schema import DISPUTE_FRAME_COLUMNS, LABEL_COLUMN
from eval.auc import auc_batch

CEILING_FRACTION = 0.98
"""A feature is flagged when `|auc - 0.5| >= CEILING_FRACTION * |bayes_auc - 0.5|`.

0.98 is only defensible if real features sit well below it, so here is the
measured margin rather than an assurance (n=5,000, seed 11): the Bayes AUC is
**0.7397** (lift 0.2397), and the strongest legitimate feature,
`ip_geo_billing_distance_km`, reaches **69.9%** of that ceiling. So the
threshold could be set anywhere from about 0.71 to 0.99 without changing a
single verdict — roughly 28 points of headroom.

That is real headroom but it is *not* enormous, and an earlier draft of this
comment guessed "roughly a third of the ceiling" before measuring, which was
wrong by a factor of two. `tests/test_generator_leakage_guard.py` now asserts
the margin directly, so a generator change that narrows it fails loudly and
this constant stops being a round number and starts needing an argument.
"""

MIN_ABSOLUTE_LIFT = 0.02
"""Guard (b) additionally requires `|auc - 0.5| >= MIN_ABSOLUTE_LIFT` before it
flags anything.

Found by the shuffled-label control, which the relative test alone failed.
With the label shuffled the Bayes AUC collapses to ~0.4972 - a lift of 0.003 -
so every feature's residual sampling noise trivially exceeded 98% of it and
seven legitimate features were flagged. A ratio against a denominator that is
itself indistinguishable from zero is not a measurement.

The floor makes the guard say the true thing: when no ranking of `p` can
separate the classes, no feature reaching "98% of nothing" is evidence of
anything. A leak under a shuffled label is still caught - by guard (a)'s hash
check, which needs no signal at all. That division of labour is why there are
three guards rather than one.

The floor can only ever *suppress* a flag, never create one, so the risk it
carries is a missed leak rather than a false alarm. It is set at 0.02 because
98% of the real Bayes lift is 0.2349 — two orders of magnitude above it — so on
unshuffled data no column can be near the relative threshold and under the
floor at the same time, and the floor is inert. (Several legitimate features do
sit below 0.02 in absolute lift: `prior_dispute_count` at 0.0027,
`filed_at`/`respond_by` at 0.0049, `delivery_confirmed` at 0.0063. They are
nowhere near the relative threshold either, so the floor changes nothing about
them.)"""

MAX_NORMALIZED_MUTUAL_INFORMATION = 0.30
"""Flag an object column whose normalised MI with the label exceeds this.

A perfect string leak scores 1.0. Measured on this generator (n=5,000, seed
11) the legitimate categoricals score `card_network` 0.000463, `reason_code`
0.000155, and exactly 0 for the three constant columns. Three orders of
magnitude of headroom, so the constant is not doing fine-grained work; it is
set well clear of the noise floor rather than tuned.

This headroom only exists because timestamps are routed to guard (b) instead
(see `_as_ordered_values`). While they were treated as categorical they scored
0.1201 — not a leak, an artefact of a near-unique column's own cardinality —
which would have left this threshold with far less room than it appears to
have."""

MIN_LEVEL_ROWS_FOR_PURITY = 30
"""Levels smaller than this are ignored by the purity check.

A level with one row is pure by arithmetic necessity, not by leakage — without
a floor, every high-cardinality column would be flagged. 30 is the
conventional "large enough that all-one-class is surprising" size; at the
generator's ~0.25 label prevalence, 30 consecutive same-class rows has
probability under 1e-4 by chance."""

IDENTIFIER_COLUMNS: frozenset[str] = frozenset({"id", "payment_id"})
"""Excluded from the categorical checks, and why.

Both are unique per row and assigned from the row index alone
(`assign_schema_fields`), never from a latent or the label — so every level
has exactly one row and would trip a purity check that had no row floor. They
are also excluded from the model's feature set entirely
(`disputedesk.features.build`). They are still covered by guards (a) and (b):
the allowlist checks their presence and the hash check would still catch one
being overwritten with a latent."""

FREE_TEXT_COLUMNS: frozenset[str] = frozenset({"customer_communication_log"})
"""Excluded from the categorical checks, and why — read this one carefully,
because excluding a column from a leak check is exactly the move that hides a
defect.

`customer_communication_log` is **not a model input**: `features/build.py`
excludes it explicitly, and CLAUDE.md forbids anything outside `evidence/`
from treating this text as a feature source. It is evidence text handed to the
LLM, and it is *designed* to carry signal about `true_fraud` (GENERATOR.md §3)
— that is the point of it. Running a leak check against it would flag the
generator working as documented.

The signal it carries is not unmeasured: `eval/extraction_comparison.py`
measures exactly that, and reports it as a result rather than suppressing it.
Like the identifiers above, this column is still covered by guards (a) and
(b)."""


class LeakageError(AssertionError):
    """Raised when a feature column is derivable from the label or a latent.

    Subclasses `AssertionError` so the guard reads as an assertion at its call
    sites and so a bare `pytest.raises(AssertionError)` written against the
    old guard still behaves."""


@dataclass(frozen=True)
class LeakageReport:
    """Everything the guard measured, whether or not it fired.

    Returned separately from `assert_no_leakage` so the margins are
    inspectable: a guard whose thresholds cannot be checked against the actual
    distribution of feature scores is a guard nobody can calibrate.
    """

    bayes_auc: float
    univariate_auc: dict[str, float] = field(default_factory=dict)
    normalized_mutual_information: dict[str, float] = field(default_factory=dict)
    pure_levels: dict[str, list[str]] = field(default_factory=dict)
    column_hashes: dict[str, str] = field(default_factory=dict)
    latent_hashes: dict[str, str] = field(default_factory=dict)

    @property
    def ceiling_lift(self) -> float:
        return abs(self.bayes_auc - 0.5)

    def lift_fraction(self, column: str) -> float:
        """How far up the Bayes ceiling a column reaches, as a fraction. 1.0
        means it discriminates as well as knowing `p` itself."""
        if self.ceiling_lift == 0.0:
            return 0.0
        return abs(self.univariate_auc[column] - 0.5) / self.ceiling_lift


def _value_hash(series: pd.Series) -> str:
    """A hash of a column's values, normalised so that an exact copy hashes
    identically regardless of the two columns' declared dtypes.

    Numeric and boolean columns are canonicalised to float64 bytes, so a bool
    latent copied into an int column still matches. Everything else is hashed
    as its string form.
    """
    if _is_ordered(series):
        payload = np.ascontiguousarray(_as_ordered_values(series)).tobytes()
    else:
        payload = "\x00".join(series.astype(str)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _univariate_auc(values: np.ndarray, label: np.ndarray) -> float:
    return float(auc_batch(label.astype(int)[None, :], values.astype(float)[None, :])[0])


def _pure_levels(values: pd.Series, label: np.ndarray, min_level_rows: int) -> list[str]:
    frame = pd.DataFrame({"level": values.astype(str), "label": label.astype(int)})
    grouped = frame.groupby("level")["label"].agg(["count", "mean"])
    big_enough = grouped[grouped["count"] >= min_level_rows]
    pure = big_enough[(big_enough["mean"] == 0.0) | (big_enough["mean"] == 1.0)]
    return sorted(pure.index.tolist())


def _as_ordered_values(series: pd.Series) -> np.ndarray:
    """Numeric, boolean and datetime columns as floats a rank statistic can
    order. Timestamps go through guard (b) rather than guard (c): they are
    ordered quantities, so univariate AUC is both well-defined on them and a
    far stronger check than mutual information, which on a near-unique column
    mostly measures the column's own cardinality.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.to_numpy(dtype="datetime64[ns]").astype("int64").astype(float)
    return series.to_numpy(dtype=float)


def _is_ordered(series: pd.Series) -> bool:
    return (
        pd.api.types.is_numeric_dtype(series)
        or pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_datetime64_any_dtype(series)
    )


def _numeric_and_object_columns(features_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric, categorical = [], []
    for column in features_df.columns:
        if column == LABEL_COLUMN or column in IDENTIFIER_COLUMNS:
            continue
        series = features_df[column]
        if _is_ordered(series):
            numeric.append(column)
        elif column not in FREE_TEXT_COLUMNS:
            categorical.append(column)
    return numeric, categorical


def leakage_report(
    features_df: pd.DataFrame,
    latents_df: pd.DataFrame | None,
    check_allowlist: bool = True,
    min_level_rows: int = MIN_LEVEL_ROWS_FOR_PURITY,
) -> LeakageReport:
    """Measure every quantity the three guards test, without raising.

    `latents_df` may be `None` for a hand-built frame with no generator behind
    it; the Bayes ceiling then falls back to the label's own perfect ranking
    (AUC 1.0), which makes guard (b) maximally permissive - appropriate,
    because with no `p` available there is no ceiling to measure against and
    the honest thing is to let guards (a) and (c) carry the check.
    `check_allowlist` is accepted for signature symmetry with
    `assert_no_leakage` and does not affect what is measured.
    """
    del check_allowlist
    label = features_df[LABEL_COLUMN].to_numpy().astype(int)
    bayes_auc = (
        _univariate_auc(latents_df["p"].to_numpy(), label) if latents_df is not None else 1.0
    )

    numeric, categorical = _numeric_and_object_columns(features_df)
    return LeakageReport(
        bayes_auc=bayes_auc,
        univariate_auc={
            c: _univariate_auc(_as_ordered_values(features_df[c]), label) for c in numeric
        },
        normalized_mutual_information={
            c: float(normalized_mutual_info_score(features_df[c].astype(str), label))
            for c in categorical
        },
        pure_levels={
            c: levels
            for c in categorical
            if (levels := _pure_levels(features_df[c], label, min_level_rows))
        },
        column_hashes={
            c: _value_hash(features_df[c]) for c in features_df.columns if c != LABEL_COLUMN
        },
        latent_hashes=(
            {c: _value_hash(latents_df[c]) for c in latents_df.columns if c != "id"}
            if latents_df is not None
            else {}
        ),
    )


def _allowlist_problems(features_df: pd.DataFrame) -> list[str]:
    actual = set(features_df.columns)
    unexpected = sorted(actual - DISPUTE_FRAME_COLUMNS)
    missing = sorted(DISPUTE_FRAME_COLUMNS - actual)
    problems = []
    if unexpected:
        problems.append(
            f"(a) provenance: undeclared column(s) {unexpected} in the feature frame - "
            "every column must be declared in DisputeRecord and reviewed"
        )
    if missing:
        problems.append(f"(a) provenance: declared column(s) {missing} missing from the frame")
    return problems


def _hash_problems(report: LeakageReport) -> list[str]:
    latent_by_hash = {digest: name for name, digest in report.latent_hashes.items()}
    return [
        f"(a) provenance: feature column '{column}' has values identical to latent "
        f"'{latent_by_hash[digest]}' - an exact copy, regardless of what it correlates with"
        for column, digest in report.column_hashes.items()
        if digest in latent_by_hash
    ]


def _ceiling_problems(
    report: LeakageReport, ceiling_fraction: float, min_absolute_lift: float
) -> list[str]:
    """Both conditions must hold: the feature reaches `ceiling_fraction` of the
    Bayes lift *and* it discriminates at all in absolute terms. See
    `MIN_ABSOLUTE_LIFT` for why the second is not redundant.
    """
    return [
        f"(b) discrimination ceiling: feature '{column}' reaches "
        f"{report.lift_fraction(column):.3f} of the Bayes lift "
        f"(auc {auc:.4f} vs bayes {report.bayes_auc:.4f}) - no legitimate noisy "
        "reading of a latent can discriminate as well as the latent itself"
        for column, auc in report.univariate_auc.items()
        if report.lift_fraction(column) >= ceiling_fraction and abs(auc - 0.5) >= min_absolute_lift
    ]


def _categorical_problems(report: LeakageReport, max_nmi: float) -> list[str]:
    problems = [
        f"(c) categorical: column '{column}' has normalised mutual information "
        f"{nmi:.4f} with the label"
        for column, nmi in report.normalized_mutual_information.items()
        if nmi > max_nmi
    ]
    problems.extend(
        f"(c) categorical: column '{column}' has level(s) {levels} with at least "
        f"{MIN_LEVEL_ROWS_FOR_PURITY} rows and a label rate of exactly 0 or 1"
        for column, levels in report.pure_levels.items()
    )
    return problems


def assert_no_leakage(
    features_df: pd.DataFrame,
    latents_df: pd.DataFrame | None,
    check_allowlist: bool = True,
    ceiling_fraction: float = CEILING_FRACTION,
    min_absolute_lift: float = MIN_ABSOLUTE_LIFT,
    max_nmi: float = MAX_NORMALIZED_MUTUAL_INFORMATION,
    min_level_rows: int = MIN_LEVEL_ROWS_FOR_PURITY,
) -> LeakageReport:
    """Run all three guards and raise `LeakageError` naming every problem
    found, not just the first.

    Returns the report on success so a caller that wants the margins does not
    have to measure twice. `check_allowlist=False` skips guard (a)'s set
    equality only (not its hash check) - for tests that deliberately add a
    column to exercise guards (b) or (c) in isolation.
    """
    report = leakage_report(features_df, latents_df, min_level_rows=min_level_rows)

    problems: list[str] = []
    if check_allowlist:
        problems.extend(_allowlist_problems(features_df))
    problems.extend(_hash_problems(report))
    problems.extend(_ceiling_problems(report, ceiling_fraction, min_absolute_lift))
    problems.extend(_categorical_problems(report, max_nmi))

    if problems:
        raise LeakageError("leakage detected:\n  " + "\n  ".join(problems))
    return report
