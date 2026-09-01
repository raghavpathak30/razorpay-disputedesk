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


def test_precision_is_non_decreasing_in_cost():
    # A higher cost raises the per-row breakeven p_win required to contest
    # (expected_value = p_win * amount - cost > 0), so the policy contests a
    # narrower, higher-confidence slice as cost rises - precision should not
    # fall.
    _per_seed, summary = _run()
    sorted_by_cost = summary.sort_values("representment_cost_inr")
    assert sorted_by_cost["precision_median"].is_monotonic_increasing


def test_recall_is_non_increasing_in_cost():
    # The same narrowing slice means recall (of the winnable disputes that
    # get contested) should not rise as cost rises.
    _per_seed, summary = _run()
    sorted_by_cost = summary.sort_values("representment_cost_inr")
    assert sorted_by_cost["recall_median"].is_monotonic_decreasing


def test_precision_recall_in_unit_range():
    per_seed, _summary = _run()
    assert per_seed["policy_precision"].between(0, 1).all()
    assert per_seed["policy_recall"].between(0, 1).all()


def test_escalate_rate_is_invariant_to_cost_per_seed():
    # decide()'s low-confidence check runs before the cost-dependent
    # expected_value branch and never reads cost or amount, so which rows
    # escalate is fixed by p_win and low_confidence_band alone - both held
    # fixed across this sweep. The escalate rate should therefore be
    # identical across every swept cost, within one seed.
    per_seed, _summary = _run()
    for seed in CI_SEEDS:
        rates = per_seed.loc[per_seed["seed"] == seed, "policy_escalate_rate"]
        assert rates.nunique() == 1


def test_escalate_rate_in_unit_range():
    per_seed, _summary = _run()
    assert per_seed["policy_escalate_rate"].between(0, 1).all()
