"""Exact-value regression on the paired cost sweep (remediation defect 1.1).

**These numbers are not results.** They are a fixed small run - 8 seeds,
5,000 rows, three cost points - committed to four decimal places so that any
change to the generator, the model, the policy engine, or the estimator shows
up as a failing test rather than as a quietly different README. The headline
sweep is 20 seeds x 15,000 rows and lives in `DECISIONS.md`; at this scale the
paired advantage at ₹200 is *negative*, which is the opposite of the headline
finding and exactly why nothing here may be quoted.

Regenerate the committed values only when a change to the pipeline is
intended, and record why in `DECISIONS.md` when you do:

    python -c "
    from eval.cost_sensitivity import summarize_sweep, sweep_representment_cost
    from eval.harness import fixed_seed_set
    print(summarize_sweep(sweep_representment_cost(fixed_seed_set(8), 5000, [50.,200.,400.])))
    "
"""

import pytest

from eval.cost_sensitivity import summarize_sweep, sweep_representment_cost
from eval.harness import fixed_seed_set

CI_SEEDS = fixed_seed_set(8, start=0)
CI_N_ROWS = 5000
CI_COSTS = [50.0, 200.0, 400.0]

# cost -> (paired mean, paired median, ci low, ci high, n positive)
COMMITTED_PAIRED_ADVANTAGE = {
    50.0: (-112.5258, 142.9187, -534.8206, 251.2606, 5),
    200.0: (-3596.1703, -1994.2882, -7515.7543, -268.0576, 2),
    400.0: (3125.1850, 1938.8082, -7742.7972, 14196.0185, 5),
}

# cost -> (precision median, recall median, escalate rate median)
COMMITTED_POLICY_RATES = {
    50.0: (0.2453, 0.9984, 0.0814),
    200.0: (0.2555, 0.9641, 0.0814),
    400.0: (0.2666, 0.8887, 0.0814),
}


@pytest.fixture(scope="module")
def summary():
    per_seed = sweep_representment_cost(CI_SEEDS, CI_N_ROWS, CI_COSTS)
    return summarize_sweep(per_seed).set_index("representment_cost_inr")


@pytest.mark.parametrize("cost", sorted(COMMITTED_PAIRED_ADVANTAGE))
def test_paired_advantage_matches_the_committed_values(cost, summary):
    mean, median, ci_low, ci_high, n_positive = COMMITTED_PAIRED_ADVANTAGE[cost]
    row = summary.loc[cost]

    assert row["advantage_paired_mean"] == pytest.approx(mean, abs=1e-4)
    assert row["advantage_paired_median"] == pytest.approx(median, abs=1e-4)
    assert row["advantage_ci_low"] == pytest.approx(ci_low, abs=1e-4)
    assert row["advantage_ci_high"] == pytest.approx(ci_high, abs=1e-4)
    assert int(row["advantage_n_positive"]) == n_positive
    assert int(row["n_seeds"]) == len(CI_SEEDS)


@pytest.mark.parametrize("cost", sorted(COMMITTED_POLICY_RATES))
def test_policy_rates_match_the_committed_values(cost, summary):
    precision, recall, escalate_rate = COMMITTED_POLICY_RATES[cost]
    row = summary.loc[cost]

    assert row["precision_median"] == pytest.approx(precision, abs=1e-4)
    assert row["recall_median"] == pytest.approx(recall, abs=1e-4)
    assert row["escalate_rate_median"] == pytest.approx(escalate_rate, abs=1e-4)


def test_the_bootstrap_interval_is_reproducible_across_runs(summary):
    """The committed CI bounds above are only meaningful if the bootstrap is
    deterministic - otherwise this whole file would flake.
    """
    rerun = summarize_sweep(
        sweep_representment_cost(CI_SEEDS, CI_N_ROWS, CI_COSTS)
    ).set_index("representment_cost_inr")

    for cost in CI_COSTS:
        assert rerun.loc[cost, "advantage_ci_low"] == summary.loc[cost, "advantage_ci_low"]
        assert rerun.loc[cost, "advantage_ci_high"] == summary.loc[cost, "advantage_ci_high"]


def test_the_escalate_rate_is_invariant_to_representment_cost(summary):
    """`decide()`'s low-confidence check runs before the cost-dependent
    branch and reads neither cost nor amount, so which rows escalate cannot
    depend on the swept cost. Measured rather than assumed - a change that
    made escalation cost-dependent would silently couple the abstention path
    to the cost sweep.
    """
    rates = set(summary["escalate_rate_median"].round(10))
    assert len(rates) == 1
