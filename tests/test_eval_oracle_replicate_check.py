"""GENERATOR.md §5's own specified cross-check for the closed-form oracle
sweep: "This can be cross-checked empirically by drawing many replicate
label-samples per p-bucket and confirming the closed-form curve matches; that
check belongs in Phase 2's tests, not here." This is that test.

Why this test exists, specifically: a Phase 1 sanity check measured
`average_precision_score(y_true, p_true)` on the seed-42 holdout at 0.4335,
from one realized `Bernoulli(p)` label draw. Phase 2's harness instead reports
the closed-form `oracle_pr_auc(p_true)`, which came out to 0.4572 (median
across 20 seeds) - outside a naive read of "the" ceiling from the single
seed-42 draw. Two different quantities are in play:

- `oracle_pr_auc(p)` is a closed-form *expectation* of average precision over
  infinitely many repeated `Bernoulli(p)` label draws, holding `p` fixed.
- `average_precision_score(y_true, p_true)` on one realized `y_true` is a
  single noisy sample of that same random quantity.

The hypothesis - not assumed, checked below - is that the closed form equals
the *mean* of the realized metric over many replicate draws, and that a
single draw (like Phase 1's) is expected to land some distance from that mean
purely from sampling variance, not from a formula disagreement.
"""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from disputedesk.generator.config import GeneratorConfig
from disputedesk.generator.pipeline import generate_dataset, temporal_split
from eval.oracle import oracle_pr_auc

N_ROWS = 15000
SEED = 42
N_REPLICATES = 500


@pytest.fixture(scope="module")
def replicate_study():
    """One holdout p-vector (seed 42, matching Phase 1's check), the
    closed-form oracle PR-AUC on it, and N_REPLICATES realized average
    precision scores from independent Bernoulli(p) label draws on that same
    p-vector.
    """
    config = GeneratorConfig()
    features_df, debug_df = generate_dataset(N_ROWS, SEED, config)
    _train_df, test_df, _boundary = temporal_split(features_df, config)
    p_holdout = debug_df.loc[test_df.index, "p"].to_numpy()

    closed_form = oracle_pr_auc(p_holdout)

    rng = np.random.default_rng(0)
    replicate_ap = np.empty(N_REPLICATES)
    for i in range(N_REPLICATES):
        y_replicate = rng.random(p_holdout.shape[0]) < p_holdout
        replicate_ap[i] = average_precision_score(y_replicate, p_holdout)

    return {
        "p_holdout": p_holdout,
        "closed_form": closed_form,
        "replicate_ap": replicate_ap,
    }


def test_closed_form_equals_the_mean_of_replicate_label_draws(replicate_study):
    """The actual check GENERATOR.md §5 specifies. Agreement is judged in
    standard errors of the replicate mean, not an arbitrary absolute
    tolerance - the right bar for "does the closed form estimate the same
    quantity the replicate mean estimates."
    """
    closed_form = replicate_study["closed_form"]
    replicate_ap = replicate_study["replicate_ap"]

    replicate_mean = replicate_ap.mean()
    replicate_std = replicate_ap.std(ddof=1)
    standard_error = replicate_std / np.sqrt(N_REPLICATES)

    gap_in_standard_errors = abs(closed_form - replicate_mean) / standard_error

    assert gap_in_standard_errors < 4, (
        f"closed-form oracle PR-AUC ({closed_form:.4f}) disagrees with the mean of "
        f"{N_REPLICATES} replicate label draws ({replicate_mean:.4f}, std {replicate_std:.4f}, "
        f"SE {standard_error:.4f}) by {gap_in_standard_errors:.1f} standard errors - "
        "they may be estimating different quantities, not just differing by sampling noise."
    )


def test_a_single_realized_draw_is_within_a_few_standard_deviations_of_the_mean(replicate_study):
    """Confirms (does not assume) the hypothesis that a single realized draw -
    like Phase 1's seed-42 average_precision_score(y_true, p_true) = 0.4335 -
    is ordinary sampling noise around the closed-form value, not evidence the
    two formulas disagree. A single draw landing within a handful of standard
    deviations of the replicate distribution is consistent with "noisy sample
    of the same quantity"; landing many std devs away would falsify it.
    """
    closed_form = replicate_study["closed_form"]
    replicate_ap = replicate_study["replicate_ap"]
    replicate_std = replicate_ap.std(ddof=1)

    single_draw_rng = np.random.default_rng(1)
    p_holdout = replicate_study["p_holdout"]
    y_single = single_draw_rng.random(p_holdout.shape[0]) < p_holdout
    single_draw_ap = average_precision_score(y_single, p_holdout)

    gap_in_std_devs = abs(single_draw_ap - closed_form) / replicate_std
    assert gap_in_std_devs < 5, (
        f"a single realized draw's AP ({single_draw_ap:.4f}) is {gap_in_std_devs:.1f} standard "
        f"deviations (std={replicate_std:.4f}) from the closed-form value ({closed_form:.4f}) - "
        "too far to attribute to single-draw sampling noise alone."
    )
