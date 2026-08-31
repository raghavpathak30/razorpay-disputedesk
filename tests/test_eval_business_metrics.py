"""Hand-built cases for the rupee metrics (SPEC.md §6): recovered rupees per
decision path, both baselines, and FP/FN cost accounting.
"""

import numpy as np

from disputedesk.policy.config import PolicyConfig
from disputedesk.policy.engine import Decision
from eval.business_metrics import (
    accept_everything_recovered,
    build_business_row,
    contest_everything_recovered,
    decide_batch,
    escalated_amount_share,
    escalated_summary,
    false_negative_cost,
    false_positive_cost,
    recovered_rupees,
)

COST = 400.0


def test_recovered_rupees_for_accept_is_zero_regardless_of_true_label():
    decisions = np.array([Decision.ACCEPT, Decision.ACCEPT], dtype=object)
    won = np.array([True, False])
    amount = np.array([5000.0, 5000.0])
    recovered = recovered_rupees(decisions, won, amount, COST)
    assert list(recovered) == [0.0, 0.0]


def test_recovered_rupees_for_contest_and_won_is_amount_minus_cost():
    decisions = np.array([Decision.CONTEST], dtype=object)
    won = np.array([True])
    amount = np.array([5000.0])
    recovered = recovered_rupees(decisions, won, amount, COST)
    assert recovered[0] == 5000.0 - COST


def test_recovered_rupees_for_contest_and_lost_is_negative_cost_only():
    decisions = np.array([Decision.CONTEST], dtype=object)
    won = np.array([False])
    amount = np.array([5000.0])
    recovered = recovered_rupees(decisions, won, amount, COST)
    assert recovered[0] == -COST


def test_recovered_rupees_for_escalate_is_zero_under_the_default_zero_mode():
    decisions = np.array([Decision.ESCALATE], dtype=object)
    won = np.array([True])
    amount = np.array([5000.0])
    recovered = recovered_rupees(decisions, won, amount, COST)
    assert recovered[0] == 0.0


def test_recovered_rupees_for_escalate_under_oracle_mode_credits_the_best_action():
    decisions = np.array([Decision.ESCALATE, Decision.ESCALATE], dtype=object)
    won = np.array([True, False])
    amount = np.array([5000.0, 5000.0])
    recovered = recovered_rupees(decisions, won, amount, COST, escalate_mode="oracle")
    # won=True -> as if contested and won; won=False -> as if accepted (0), not
    # -COST - the oracle never makes the losing choice.
    assert list(recovered) == [5000.0 - COST, 0.0]


def test_recovered_rupees_for_escalate_under_naive_contest_mode_matches_contesting():
    decisions = np.array([Decision.ESCALATE, Decision.ESCALATE], dtype=object)
    won = np.array([True, False])
    amount = np.array([5000.0, 5000.0])
    recovered = recovered_rupees(decisions, won, amount, COST, escalate_mode="naive_contest")
    assert list(recovered) == [5000.0 - COST, -COST]


def test_non_escalated_rows_are_unaffected_by_escalate_mode():
    decisions = np.array([Decision.CONTEST, Decision.ACCEPT], dtype=object)
    won = np.array([True, False])
    amount = np.array([5000.0, 5000.0])
    for mode in ("zero", "oracle", "naive_contest"):
        recovered = recovered_rupees(decisions, won, amount, COST, escalate_mode=mode)
        assert list(recovered) == [5000.0 - COST, 0.0]


def test_accept_everything_baseline_is_always_zero():
    recovered = accept_everything_recovered(5)
    assert list(recovered) == [0.0] * 5


def test_contest_everything_baseline_matches_per_row_contest_math():
    won = np.array([True, False])
    amount = np.array([1000.0, 2000.0])
    recovered = contest_everything_recovered(won, amount, COST)
    assert list(recovered) == [1000.0 - COST, -COST]


def test_decide_batch_matches_scalar_decide():
    config = PolicyConfig(representment_cost_inr=COST, low_confidence_band=(0.45, 0.55))
    p_win = np.array([0.9, 0.1, 0.5])
    amount = np.array([5000.0, 1000.0, 5000.0])
    decisions = decide_batch(p_win, amount, config)
    assert list(decisions) == [Decision.CONTEST, Decision.ACCEPT, Decision.ESCALATE]


def test_false_positive_cost_counts_contested_and_lost_only():
    decisions = np.array(
        [Decision.CONTEST, Decision.CONTEST, Decision.ACCEPT, Decision.ESCALATE], dtype=object
    )
    won = np.array([False, True, False, False])
    count, total = false_positive_cost(decisions, won, COST)
    assert count == 1
    assert total == COST


def test_false_negative_cost_counts_accepted_and_winnable_only():
    decisions = np.array(
        [Decision.ACCEPT, Decision.ACCEPT, Decision.CONTEST, Decision.ESCALATE], dtype=object
    )
    won = np.array([True, False, True, True])
    amount = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    count, total = false_negative_cost(decisions, won, amount)
    assert count == 1
    assert total == 1000.0


def test_escalated_summary_counts_and_sums_escalated_only():
    decisions = np.array([Decision.ESCALATE, Decision.CONTEST, Decision.ESCALATE], dtype=object)
    amount = np.array([1000.0, 2000.0, 3000.0])
    count, total = escalated_summary(decisions, amount)
    assert count == 2
    assert total == 4000.0


def test_escalated_amount_share_is_fraction_of_total_holdout_rupees():
    decisions = np.array([Decision.ESCALATE, Decision.CONTEST, Decision.ACCEPT], dtype=object)
    amount = np.array([2000.0, 3000.0, 5000.0])
    share = escalated_amount_share(decisions, amount)
    assert share == 2000.0 / 10000.0


def test_escalated_amount_share_can_exceed_the_escalated_count_share():
    # Two of five disputes (40% by count) escalated, but they carry most of
    # the money (80% by amount) - the count alone would understate this.
    decisions = np.array(
        [Decision.ESCALATE, Decision.ESCALATE, Decision.CONTEST, Decision.ACCEPT, Decision.ACCEPT],
        dtype=object,
    )
    amount = np.array([4000.0, 4000.0, 500.0, 500.0, 1000.0])
    share = escalated_amount_share(decisions, amount)
    assert share == 8000.0 / 10000.0


def test_build_business_row_is_internally_consistent_on_a_hand_built_set():
    # Four disputes, hand-chosen so each decision path fires once:
    #   p_win=0.9 -> contest, won=True  -> recovered = amount - cost
    #   p_win=0.1 -> contest? no: EV = 0.1*2000-400 = -200 -> accept, won=False (true negative)
    #   p_win=0.9 -> contest, won=False -> recovered = -cost (a false positive)
    #   p_win=0.5 -> escalate (band), won=True (a "would-be" false negative if forced to accept)
    config = PolicyConfig(representment_cost_inr=COST, low_confidence_band=(0.45, 0.55))
    p_win = np.array([0.9, 0.1, 0.9, 0.5])
    won = np.array([True, False, False, True])
    amount = np.array([5000.0, 2000.0, 3000.0, 4000.0])

    row = build_business_row(p_win, won, amount, config)

    assert row["n"] == 4
    # zero mode: escalated row (index 3, won=True) credited 0.
    expected_total_recovered_zero = (5000.0 - COST) + 0.0 + (-COST) + 0.0
    assert row["policy_recovered_per_1000_inr"] == expected_total_recovered_zero / 4 * 1000.0
    # oracle mode: same escalated row credited amount-cost (it was won).
    expected_total_recovered_oracle = (5000.0 - COST) + 0.0 + (-COST) + (4000.0 - COST)
    assert (
        row["policy_recovered_per_1000_inr_escalate_oracle"]
        == expected_total_recovered_oracle / 4 * 1000.0
    )
    # naive_contest mode: same as oracle here since the escalated row's true
    # label happens to be a win too - both credit amount-cost for it.
    assert (
        row["policy_recovered_per_1000_inr_escalate_naive_contest"]
        == expected_total_recovered_oracle / 4 * 1000.0
    )
    assert row["false_positive_count"] == 1
    assert row["false_positive_cost_per_1000_inr"] == COST / 4 * 1000.0
    assert row["false_negative_count"] == 0
    assert row["false_negative_cost_per_1000_inr"] == 0.0
    assert row["escalated_count"] == 1
    assert row["escalated_amount_per_1000_inr"] == 4000.0 / 4 * 1000.0
    assert row["escalated_amount_share_of_holdout"] == 4000.0 / (5000.0 + 2000.0 + 3000.0 + 4000.0)

    baseline_a = contest_everything_recovered(won, amount, COST).sum() / 4 * 1000.0
    assert row["baseline_a_contest_everything_recovered_per_1000_inr"] == baseline_a
    assert row["baseline_b_accept_everything_recovered_per_1000_inr"] == 0.0
