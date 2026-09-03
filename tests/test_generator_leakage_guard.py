"""The leakage guard (remediation defect 2.1) — CLAUDE.md's "first-class test,
not a script", rebuilt because the previous version passed on a frame that
contained an exact copy of the generator's `p` *and* a perfect string leak.

**What was wrong.** The old guard was a single check: Pearson `|r| > 0.9`
between each numeric column and the target. Correlation against a *binary*
label cannot work as a leak detector, because its ceiling is set by the
label's own noise rather than by how much the feature knows. `won_if_contested`
is `Bernoulli(p)`, so a verbatim copy of `p` — a total leak, the model would
be reading the answer — correlates only ~0.36 with it. The guard's threshold
was 0.9. It could not fire on the worst possible leak.

Three independent guards replace it, each catching what the others miss:

- **(a) provenance** — set equality against a frozen column allowlist, plus a
  value-hash comparison against every latent column. Catches an exact copy
  with no statistics at all, so it holds regardless of how noisy the label is.
- **(b) discrimination ceiling** — univariate AUC against the *Bayes* AUC
  achievable by ranking on true `p`. This is the right yardstick precisely
  because it is the label-noise-adjusted one: a copy of `p` sits exactly at
  the ceiling; a legitimate feature cannot reach it.
- **(c) categorical** — normalised mutual information, plus a per-level label
  purity check. Correlation is meaningless for object columns; the old guard
  simply skipped them, which is how a perfect string leak got through.

The last two fixtures below are controls. A guard that fires on everything is
as useless as one that fires on nothing, so "legitimate features" and
"legitimate features + shuffled label" must both pass.
"""

import numpy as np
import pandas as pd
import pytest

from disputedesk.generator.config import GeneratorConfig
from disputedesk.generator.pipeline import generate_dataset
from eval.leakage import LeakageError, assert_no_leakage, leakage_report


@pytest.fixture(scope="module")
def generated():
    features_df, debug_df = generate_dataset(5000, seed=11, config=GeneratorConfig())
    return features_df, debug_df


@pytest.fixture(scope="module")
def report(generated):
    features_df, debug_df = generated
    return leakage_report(features_df, debug_df)


# --------------------------------------------------------------------------
# The red-team fixture set
# --------------------------------------------------------------------------


def test_legitimate_features_pass(generated):
    """Control 1. The generator's real output must not trip any of the three
    guards — otherwise the guard is measuring its own strictness.
    """
    features_df, debug_df = generated
    assert_no_leakage(features_df, debug_df)


def test_legitimate_features_with_a_shuffled_label_pass(generated):
    """Control 2, and the sharper of the two. Destroying the feature/label
    relationship entirely must not make the guard fire: a guard that fires
    here is keying on something other than leakage.
    """
    features_df, debug_df = generated
    shuffled = features_df.copy()
    rng = np.random.default_rng(0)
    shuffled["won_if_contested"] = rng.permutation(shuffled["won_if_contested"].to_numpy())

    assert_no_leakage(shuffled, debug_df)


def test_a_verbatim_copy_of_p_raises(generated):
    """The fixture the old guard passed. `p` correlates only ~0.36 with the
    Bernoulli(p) label, so no correlation threshold that also admits real
    features could ever have caught this.
    """
    features_df, debug_df = generated
    leaky = features_df.copy()
    leaky["copied_p"] = debug_df["p"].to_numpy()

    with pytest.raises(LeakageError, match="copied_p"):
        assert_no_leakage(leaky, debug_df)


def test_a_perfect_string_leak_raises(generated):
    """The other fixture the old guard passed, and for a different reason: it
    skipped every non-numeric column outright.
    """
    features_df, debug_df = generated
    leaky = features_df.copy()
    leaky["outcome_note"] = np.where(
        features_df["won_if_contested"].to_numpy(), "recovered", "written off"
    )

    with pytest.raises(LeakageError, match="outcome_note"):
        assert_no_leakage(leaky, debug_df)


def test_a_direct_label_copy_raises(generated):
    features_df, debug_df = generated
    leaky = features_df.copy()
    leaky["copied_label"] = features_df["won_if_contested"].astype(float).to_numpy()

    with pytest.raises(LeakageError, match="copied_label"):
        assert_no_leakage(leaky, debug_df)


# --------------------------------------------------------------------------
# (a) provenance allowlist
# --------------------------------------------------------------------------


def test_an_unexpected_column_raises_even_if_it_is_harmless(generated):
    """The allowlist is set *equality*, not a subset check. A column nobody
    declared is a column nobody reviewed, and reviewing it is cheaper than
    discovering later that it leaked.
    """
    features_df, debug_df = generated
    extra = features_df.copy()
    extra["harmless_row_number"] = np.arange(len(extra))

    with pytest.raises(LeakageError, match="harmless_row_number"):
        assert_no_leakage(extra, debug_df)


def test_a_missing_declared_column_raises(generated):
    features_df, debug_df = generated
    truncated = features_df.drop(columns=["checkout_hour_of_day"])

    with pytest.raises(LeakageError, match="checkout_hour_of_day"):
        assert_no_leakage(truncated, debug_df)


def test_the_hash_check_catches_a_copy_that_statistics_would_miss():
    """A latent with almost no relationship to the label — `epsilon`, the
    generator's own irreducible-error term — copied verbatim into the feature
    frame. Neither AUC nor mutual information would flag it, because it
    genuinely carries no signal about the label. Provenance still must.
    """
    features_df, debug_df = generate_dataset(500, seed=3, config=GeneratorConfig())
    leaky = features_df.copy()
    leaky["ip_geo_billing_distance_km"] = debug_df["epsilon"].to_numpy()

    with pytest.raises(LeakageError, match="epsilon"):
        assert_no_leakage(leaky, debug_df, check_allowlist=False)


# --------------------------------------------------------------------------
# (b) univariate discrimination ceiling
# --------------------------------------------------------------------------


def test_the_bayes_auc_is_reported_and_beats_every_real_feature(report):
    """The margin the whole guard rests on. If a legitimate feature sat close
    to the ceiling the 98% constant would be arbitrary; the report exists so
    that margin is inspectable rather than asserted.
    """
    assert 0.5 < report.bayes_auc < 1.0
    for name, auc in report.univariate_auc.items():
        assert abs(auc - 0.5) < abs(report.bayes_auc - 0.5), name


def test_a_copy_of_p_sits_at_the_ceiling(generated):
    features_df, debug_df = generated
    leaky = features_df.copy()
    leaky["copied_p"] = debug_df["p"].to_numpy()

    report = leakage_report(leaky, debug_df)

    assert report.univariate_auc["copied_p"] == pytest.approx(report.bayes_auc, abs=1e-12)


def test_a_monotone_transform_of_p_also_sits_at_the_ceiling(generated):
    """AUC is rank-based, so a leak does not have to be a verbatim copy to be
    total. This is the case the hash check alone would miss.
    """
    features_df, debug_df = generated
    leaky = features_df.copy()
    leaky["scaled_p"] = debug_df["p"].to_numpy() * 1000.0 + 7.0

    with pytest.raises(LeakageError, match="scaled_p"):
        assert_no_leakage(leaky, debug_df, check_allowlist=False)


def test_an_inverted_copy_of_p_is_caught_too(generated):
    """`-p` ranks perfectly in the wrong direction: AUC near 0, which is just
    as much of a leak. The guard tests |auc - 0.5|, not auc.
    """
    features_df, debug_df = generated
    leaky = features_df.copy()
    leaky["negated_p"] = -debug_df["p"].to_numpy()

    with pytest.raises(LeakageError, match="negated_p"):
        assert_no_leakage(leaky, debug_df, check_allowlist=False)


# --------------------------------------------------------------------------
# (c) categorical leak check
# --------------------------------------------------------------------------


def test_a_pure_category_level_raises_even_when_mutual_information_is_low():
    """A single small-but-not-tiny level whose label rate is exactly 1.0,
    hidden among legitimate levels. Its contribution to mutual information is
    diluted by the other levels; the per-level purity check is what catches
    it.
    """
    rng = np.random.default_rng(0)
    n = 4000
    label = rng.random(n) < 0.3
    category = rng.choice(["a", "b", "c"], size=n)
    tainted = np.where(label, "winner", category)  # every "winner" row is a win

    df = pd.DataFrame({"category": tainted, "won_if_contested": label})

    report = leakage_report(df, latents_df=None, check_allowlist=False)

    assert "category" in report.pure_levels
    assert report.pure_levels["category"]


def test_a_legitimate_categorical_does_not_trip_the_purity_check(report):
    """`reason_code` and `card_network` have four levels each over 5,000 rows
    — plenty of rows per level, and no level near-pure. The control for the
    check above.
    """
    assert report.pure_levels == {}


def test_the_purity_check_ignores_levels_below_the_row_threshold():
    """A one-row level is pure by arithmetic necessity, not by leakage. The
    threshold is what stops the check from firing on every high-cardinality
    column in the frame.
    """
    rng = np.random.default_rng(1)
    n = 600
    label = rng.random(n) < 0.4
    category = np.array([f"level_{i}" for i in range(n)])  # every level has one row

    df = pd.DataFrame({"category": category, "won_if_contested": label})

    report = leakage_report(df, latents_df=None, check_allowlist=False)

    assert report.pure_levels == {}


# --------------------------------------------------------------------------
# The allowlist and the schema must not be able to drift apart
# --------------------------------------------------------------------------


def test_the_allowlist_matches_the_dispute_record_model():
    """`DISPUTE_FRAME_COLUMNS` is typed out by hand so that adding a field to
    `DisputeRecord` cannot silently widen it. This is the test that makes the
    divergence loud instead of leaving the constant to rot.
    """
    from disputedesk.generator.schema import DISPUTE_FRAME_COLUMNS, DisputeRecord

    assert DISPUTE_FRAME_COLUMNS == set(DisputeRecord.model_fields)


def test_the_latent_allowlist_matches_the_debug_record_model():
    from disputedesk.generator.schema import LATENT_FRAME_COLUMNS, DebugRecord

    assert LATENT_FRAME_COLUMNS == set(DebugRecord.model_fields) - {"id"}


def test_the_generator_emits_exactly_the_declared_columns(generated):
    features_df, debug_df = generated
    from disputedesk.generator.schema import DISPUTE_FRAME_COLUMNS, LATENT_FRAME_COLUMNS

    assert set(features_df.columns) == DISPUTE_FRAME_COLUMNS
    assert set(debug_df.columns) - {"id"} == LATENT_FRAME_COLUMNS


# --------------------------------------------------------------------------
# The margins the thresholds rest on, pinned so they cannot silently narrow
# --------------------------------------------------------------------------


def test_the_strongest_legitimate_feature_leaves_real_headroom(report):
    """The 98% constant is only a defensible round number while real features
    sit well below it. Measured: the strongest, `ip_geo_billing_distance_km`,
    reaches ~70% of the Bayes ceiling — about 28 points of headroom. If a
    generator change pushes a legitimate feature above 0.85 the constant needs
    an argument rather than a default, and this test is where that surfaces.
    """
    strongest = max(report.lift_fraction(c) for c in report.univariate_auc)

    assert strongest < 0.85, f"strongest legitimate feature reaches {strongest:.3f} of the ceiling"
    assert strongest > 0.5, "a suspiciously weak feature set - check the generator still has signal"


def test_legitimate_categoricals_are_orders_of_magnitude_below_the_nmi_threshold(report):
    from eval.leakage import MAX_NORMALIZED_MUTUAL_INFORMATION

    worst = max(report.normalized_mutual_information.values())

    assert worst < MAX_NORMALIZED_MUTUAL_INFORMATION / 100


def test_timestamps_are_checked_by_the_ceiling_guard_not_the_categorical_one(report):
    """Timestamps are ordered quantities, so univariate AUC is both defined on
    them and stronger than mutual information — which on a near-unique column
    largely measures that column's own cardinality (they scored 0.12 as
    categoricals, which would have eaten most of the NMI threshold's headroom).
    """
    for column in ("purchase_ts", "filed_at", "respond_by"):
        assert column in report.univariate_auc
        assert column not in report.normalized_mutual_information


def test_a_timestamp_that_encodes_the_label_is_caught(generated):
    """The check that routing timestamps to guard (b) actually guards them."""
    features_df, debug_df = generated
    leaky = features_df.copy()
    leaky["filed_at"] = pd.to_datetime(debug_df["p"].to_numpy() * 1e15)

    with pytest.raises(LeakageError, match="filed_at"):
        assert_no_leakage(leaky, debug_df)
