"""GENERATOR.md §5's own specified cross-check for the closed-form oracle
sweep: "This can be cross-checked empirically by drawing many replicate
label-samples per p-bucket and confirming the closed-form curve matches; that
check belongs in Phase 2's tests, not here." This is that test.

Why this test exists, specifically: a Phase 1 sanity check measured
`average_precision_score(y_true, p_true)` on the seed-42 holdout at 0.4335
(which no longer reproduces - the current generator gives 0.4305; see the
golden-fixture test below and DECISIONS.md's 2026-09-02 correction),
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


# GOLDEN FIXTURE - pinned against disputedesk.generator output at seed=42,
# n_rows=15000. If eval/generator_fingerprint.py's committed fingerprint ever
# changes, re-run and re-commit this value too (see that module's docstring).
HISTORICAL_SEED_42_REALIZED_AP = 0.4304927827841146
"""The generator's own seed-42 holdout label draw, scored against its own `p`.

This is the *golden fixture*: a value frozen from a frozen seed, reproducible
by anyone running this repository. It is deliberately not 0.4335 - see
`test_the_historical_seed_42_draw_reproduces` and DECISIONS.md's 2026-09-02
correction for why that recorded figure no longer reproduces."""


def test_the_historical_seed_42_draw_reproduces():
    """Golden-fixture regression. Asserts the committed historical value at the
    frozen seed, and draws **no** random sample of its own.

    Split out on 2026-09-02 (remediation defect 2.3). The single test that
    previously stood here claimed in its docstring to be about "Phase 1's
    seed-42 `average_precision_score(y_true, p_true)` = 0.4335" and then
    ignored it entirely: it drew a *fresh* Bernoulli sample from
    `default_rng(1)` and checked that. So the historical value was never under
    test, and the test's name and docstring described work it did not do. The
    distributional check it actually performed is worth keeping and now lives
    in `test_a_fresh_draw_is_within_a_few_standard_deviations_of_the_mean`
    below, with a docstring that says it samples.
    """
    config = GeneratorConfig()
    features_df, debug_df = generate_dataset(N_ROWS, SEED, config)
    _train_df, test_df, _boundary = temporal_split(features_df, config)
    p_holdout = debug_df.loc[test_df.index, "p"].to_numpy()
    y_holdout = test_df["won_if_contested"].to_numpy()

    realized_ap = average_precision_score(y_holdout, p_holdout)

    assert realized_ap == pytest.approx(HISTORICAL_SEED_42_REALIZED_AP, abs=1e-12)


def test_the_recorded_0_4335_no_longer_reproduces_and_the_gap_is_recorded():
    """The correction itself, pinned so it cannot be quietly re-asserted.

    DECISIONS.md's 2026-08-31 entry records 0.4335 for this quantity. Running
    the current generator at the same seed gives 0.4305 - the generator changed
    after that measurement (GENERATOR.md revision 2: the `amount` draw became
    weakly causal on `true_fraud`, and a noise feature was added), so the old
    figure describes a dataset this repository no longer produces.

    The gap is small and does not change any conclusion drawn from it - the
    single-draw-vs-closed-form reasoning holds at either value. It is recorded
    rather than silently updated because a number quoted in a docstring that
    cannot be reproduced is the exact defect class this remediation exists for.
    """
    assert abs(HISTORICAL_SEED_42_REALIZED_AP - 0.4335) < 0.005
    assert HISTORICAL_SEED_42_REALIZED_AP != pytest.approx(0.4335, abs=1e-4)


def test_a_fresh_draw_is_within_a_few_standard_deviations_of_the_mean(replicate_study):
    """Distributional sanity check. **Draws a fresh Bernoulli(p) sample** and
    asks whether one realized draw lands where the replicate distribution says
    it should.

    This is a property of the sampling distribution, not a statement about any
    historical number - which is why it is now separated from the golden
    fixture above. It confirms (does not assume) that a single realized draw is
    ordinary sampling noise around the closed-form value rather than evidence
    the two formulas disagree.
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


def test_the_historical_draw_is_also_within_the_replicate_distribution(replicate_study):
    """The claim the old test's docstring made but never checked: that the
    *historical* value - not a fresh draw - sits within ordinary sampling noise
    of the closed form.
    """
    closed_form = replicate_study["closed_form"]
    replicate_std = replicate_study["replicate_ap"].std(ddof=1)

    gap_in_std_devs = abs(HISTORICAL_SEED_42_REALIZED_AP - closed_form) / replicate_std

    assert gap_in_std_devs < 5
