"""CI-speed sanity checks for the representment_cost_inr sweep: shape,
monotonicity, and the near-cost-0 convergence between the policy and
baseline A (at cost=0 nearly everything is worth contesting, so the two
should be close). Small n_rows/seed count for CI speed, not the headline
sweep (produced by `eval/run_cost_sensitivity.py`).
"""

from disputedesk.generator.config import GeneratorConfig
from disputedesk.model.config import ModelConfig
from disputedesk.policy.config import REPRESENTMENT_COST_INR
from eval.cost_sensitivity import fixed_seed_set, summarize_sweep, sweep_representment_cost

CI_SEEDS = fixed_seed_set(4, start=0)
CI_N_ROWS = 5000
CI_COSTS = [0.0, REPRESENTMENT_COST_INR, 5000.0]


def _run():
    per_seed = sweep_representment_cost(
        CI_SEEDS, CI_N_ROWS, CI_COSTS, GeneratorConfig(), ModelConfig()
    )
    return per_seed, summarize_sweep(per_seed)


def test_sweep_produces_one_row_per_seed_per_cost():
    per_seed, _summary = _run()
    assert len(per_seed) == len(CI_SEEDS) * len(CI_COSTS)


def test_summary_has_one_row_per_cost_sorted_ascending():
    _per_seed, summary = _run()
    assert list(summary["representment_cost_inr"]) == sorted(CI_COSTS)


def test_at_cost_zero_the_policy_and_baseline_a_nearly_coincide():
    # At cost=0, expected_value = p_win * amount >= 0 whenever p_win > 0, so
    # the policy contests almost everything baseline A already contests -
    # the two should be close, not identical (the confidence band still
    # diverts some disputes, though under naive_contest scoring that only
    # matters when the *decision itself* differs, not the escalate credit).
    _per_seed, summary = _run()
    zero_cost_row = summary[summary["representment_cost_inr"] == 0.0].iloc[0]
    relative_gap = abs(zero_cost_row["policy_advantage_median"]) / max(
        abs(zero_cost_row["baseline_a_median"]), 1.0
    )
    assert relative_gap < 0.05


def test_policy_advantage_is_non_negative_at_a_high_cost():
    # At a cost well above typical dispute amounts, baseline A pays that
    # cost on every contest (including near-certain losses) while the
    # policy accepts most of them - the policy should not do worse.
    _per_seed, summary = _run()
    high_cost_row = summary[summary["representment_cost_inr"] == 5000.0].iloc[0]
    assert high_cost_row["policy_advantage_median"] >= 0


def test_baseline_a_recovered_is_non_increasing_in_cost():
    # Baseline A's math is cost * n subtracted regardless of decision
    # quality - strictly monotonic by construction, a structural sanity
    # check on the sweep mechanics themselves.
    _per_seed, summary = _run()
    sorted_by_cost = summary.sort_values("representment_cost_inr")
    assert sorted_by_cost["baseline_a_median"].is_monotonic_decreasing
