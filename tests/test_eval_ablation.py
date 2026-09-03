"""Single-feature ablation of the model (Phase 2 addendum, item B, 2026-09-03).

`ip_geo_billing_distance_km` sits at 69.9% of the Bayes discrimination
ceiling on its own (`tests/test_generator_leakage_guard.py`'s measured
margins). That is not leakage - guard (b) has a 98% flag threshold precisely
because 69.9% is comfortably below it - but a feature that strong on its own,
against a policy advantage over "contest everything" of only ≈0.66% at the
configured cost, raises a question the repository could not previously
answer: how much of the model's value is that one feature carrying, versus
the other eleven.

This module scores the business harness under three feature sets - the
single strongest feature, the top three, and the full set - using the exact
same paired estimator, seeds, and cost sweep as Phase 1
(`eval.cost_sensitivity.summarize_sweep`), so the three variants are directly
comparable to each other and to the already-reported full-feature numbers.
"""

import numpy as np
import pytest

from disputedesk.features.build import FEATURE_COLUMNS
from disputedesk.generator.config import GeneratorConfig
from disputedesk.model.config import ModelConfig
from eval.ablation import (
    TOP_1_FEATURE,
    TOP_3_FEATURES,
    predictions_for_feature_subset,
    sweep_feature_subset,
)
from eval.cost_sensitivity import summarize_sweep
from eval.harness import fixed_seed_set


def test_top_feature_constants_match_the_leakage_reports_ranking():
    """The ablation variants are supposed to be the guard's own measured
    ranking, not a hand-picked guess - this pins the constants against
    `eval.leakage`'s report on the same fixed seed the guard's own tests use,
    so the two cannot silently diverge.
    """
    from disputedesk.generator.pipeline import generate_dataset
    from eval.leakage import leakage_report

    features_df, debug_df = generate_dataset(5000, seed=11, config=GeneratorConfig())
    report = leakage_report(features_df, debug_df)

    ranked = sorted(report.univariate_auc, key=lambda c: -report.lift_fraction(c))

    assert TOP_1_FEATURE == ranked[0]
    assert TOP_3_FEATURES == tuple(ranked[:3])


def test_top_1_is_the_first_element_of_top_3():
    assert TOP_1_FEATURE == TOP_3_FEATURES[0]
    assert len(TOP_3_FEATURES) == 3


def test_predictions_for_a_restricted_feature_subset_use_only_those_columns():
    """Structural check that the restriction is real - a model "trained" on
    one feature cannot be secretly reading the others. Two calls differing
    only in a column the subset excludes must produce identical predictions.
    """
    from disputedesk.generator.pipeline import generate_dataset, temporal_split

    features_df, _debug_df = generate_dataset(3000, seed=0, config=GeneratorConfig())
    train_df, test_df, _boundary = temporal_split(features_df, GeneratorConfig())

    predicted_p = predictions_for_feature_subset(train_df, test_df, (TOP_1_FEATURE,), ModelConfig())

    tampered_test = test_df.copy()
    tampered_test["amount"] = tampered_test["amount"] * 7.0 + 1.0  # amount excluded from top-1
    predicted_p_tampered = predictions_for_feature_subset(
        train_df, tampered_test, (TOP_1_FEATURE,), ModelConfig()
    )

    assert np.array_equal(predicted_p, predicted_p_tampered)


def test_the_full_feature_variant_matches_the_unrestricted_harness():
    """The ablation's "full" variant must be the same computation as
    `eval.harness.run_seed_pipeline` - not a second, potentially-diverging
    implementation of "train on everything".
    """
    from eval.harness import run_seed_pipeline

    seed, n_rows = 0, 3000
    generator_config, model_config = GeneratorConfig(), ModelConfig()

    run = run_seed_pipeline(seed, n_rows, generator_config, model_config)
    full_predicted = predictions_for_feature_subset(
        run.train_df, run.test_df, FEATURE_COLUMNS, model_config
    )

    assert np.allclose(full_predicted, run.predicted_p)


def test_sweep_feature_subset_returns_the_same_shape_as_the_full_sweep():
    """The ablation's per-seed rows must be shaped exactly like
    `eval.cost_sensitivity.sweep_representment_cost`'s output, since
    `summarize_sweep` - the paired estimator - is reused unmodified.
    """
    seeds = fixed_seed_set(3)
    costs = [400.0]

    per_seed = sweep_feature_subset(seeds, 3000, costs, (TOP_1_FEATURE,))
    summary = summarize_sweep(per_seed)

    assert len(per_seed) == len(seeds) * len(costs)
    assert "advantage_paired_mean" in summary.columns
    assert "advantage_ci_low" in summary.columns


def test_all_three_variants_produce_a_valid_summary_at_ci_scale():
    """Deliberately does not assert an ordering between the three variants'
    advantage figures. Whether fewer features recover more or less of the
    advantage is the empirical question this module exists to answer, not
    something to bake into a test as an assumption - and at CI scale (small
    n, few seeds) it does not even hold in one consistent direction, which is
    itself informative about how noisy a 3-6-seed read is. The headline
    comparison runs at the same 20-seed, 15,000-row scale as every other
    Phase 1 number; this test only checks the three variants are each
    well-formed and independently computable.
    """
    seeds = fixed_seed_set(4)
    costs = [400.0]

    for feature_columns in ((TOP_1_FEATURE,), TOP_3_FEATURES, FEATURE_COLUMNS):
        summary = summarize_sweep(sweep_feature_subset(seeds, 3000, costs, feature_columns))
        row = summary.iloc[0]
        assert row["n_seeds"] == len(seeds)
        assert np.isfinite(row["advantage_paired_mean"])
        assert row["advantage_ci_low"] <= row["advantage_paired_mean"] <= row["advantage_ci_high"]


def test_an_unknown_feature_name_raises_rather_than_being_silently_dropped():
    with pytest.raises(ValueError, match="not a declared feature"):
        sweep_feature_subset(fixed_seed_set(1), 2000, [400.0], ("not_a_real_feature",))


# --------------------------------------------------------------------------
# GOLDEN FIXTURE - pinned against disputedesk.generator output, seeds 0-7,
# n_rows=5000. If eval/generator_fingerprint.py's committed fingerprint ever
# changes, re-run and re-commit these values too (see that module's
# docstring). The headline 20-seed x 15,000-row ablation table is in
# DECISIONS.md; this is the same computation at CI scale, frozen so the
# comparison cannot drift unnoticed. Not a claim about the direction of the
# finding at this reduced scale - see the note above about not asserting an
# ordering.
# --------------------------------------------------------------------------

COMMITTED_ABLATION_ADVANTAGE_AT_400 = {
    # variant -> (paired mean, ci low, ci high, n positive)
    "top1": (6116.0721424559015, -5077.886079946651, 17582.74497345494, 5),
    "top3": (-2288.84490758623, -14325.840202086856, 9273.733080970356, 3),
    "full": (3125.184975411801, -7742.797238000901, 14196.018475488481, 5),
}


@pytest.mark.parametrize("variant", ["top1", "top3", "full"])
def test_the_measured_ablation_advantage_at_400(variant):
    feature_columns = {
        "top1": (TOP_1_FEATURE,),
        "top3": TOP_3_FEATURES,
        "full": FEATURE_COLUMNS,
    }[variant]
    mean, ci_low, ci_high, n_positive = COMMITTED_ABLATION_ADVANTAGE_AT_400[variant]

    per_seed = sweep_feature_subset(fixed_seed_set(8), 5000, [400.0], feature_columns)
    row = summarize_sweep(per_seed).iloc[0]

    assert row["advantage_paired_mean"] == pytest.approx(mean, abs=1e-6)
    assert row["advantage_ci_low"] == pytest.approx(ci_low, abs=1e-6)
    assert row["advantage_ci_high"] == pytest.approx(ci_high, abs=1e-6)
    assert int(row["advantage_n_positive"]) == n_positive
