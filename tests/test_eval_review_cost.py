"""What the grounding gate's false-flag rate does to the cost sweep's
break-even, and the arithmetic that says so.

The property that matters: the gate's false-flag rate applies to the CONTEST
share only. An escalated dispute never reaches the gate (no evidence is
assembled on that branch) and an accepted one has no letter, so charging the
rate against the whole population would overstate the gate's cost.
"""

import math

import pytest

from eval.review_cost import (
    false_flag_rate_at_review_cost,
    human_touched_rate,
    review_cost_curve,
    review_cost_point,
)

# The measured sweep values at `representment_cost_inr=400` (seeds 0-19,
# n_rows=15000): DECISIONS.md 2026-09-02, plus the contest rate measured
# alongside this module.
ESCALATE = 0.056189
CONTEST = 0.805225
ADVANTAGE = 11210.0


class TestHumanTouchedRate:
    def test_a_zero_false_flag_rate_reproduces_the_escalate_only_rate(self):
        """The sweep's own published assumption: at a zero withheld rate the
        human-touched rate is the ESCALATE rate alone."""
        assert human_touched_rate(ESCALATE, CONTEST, 0.0) == pytest.approx(ESCALATE)

    def test_the_rate_applies_to_the_contest_share_only(self):
        touched = human_touched_rate(ESCALATE, CONTEST, 0.10)
        assert touched == pytest.approx(ESCALATE + 0.10 * CONTEST)
        # and is strictly less than charging it against everything
        assert touched < ESCALATE + 0.10

    def test_it_is_monotone_in_the_false_flag_rate(self):
        rates = [human_touched_rate(ESCALATE, CONTEST, f) for f in (0.0, 0.05, 0.2, 0.5, 1.0)]
        assert rates == sorted(rates)

    @pytest.mark.parametrize(
        "escalate,contest,flag",
        [(-0.1, 0.5, 0.1), (0.1, 1.5, 0.1), (0.1, 0.5, 1.2)],
    )
    def test_a_rate_outside_the_unit_interval_raises(self, escalate, contest, flag):
        with pytest.raises(ValueError, match="rate in"):
            human_touched_rate(escalate, contest, flag)

    def test_branches_that_sum_past_one_raise(self):
        with pytest.raises(ValueError, match="must not exceed 1"):
            human_touched_rate(0.7, 0.7, 0.1)


class TestBreakEven:
    def test_the_zero_flag_break_even_reproduces_the_recorded_figure(self):
        """DECISIONS.md 2026-09-02 records ~INR 200 at the ESCALATE rate
        alone. If this drifts, either the sweep changed or this module's
        arithmetic did, and both should be loud."""
        point = review_cost_point(0.0, ESCALATE, CONTEST, ADVANTAGE)
        assert point.break_even_review_cost_inr == pytest.approx(200.0, abs=2.0)

    def test_a_higher_false_flag_rate_lowers_the_break_even(self):
        costs = [
            review_cost_point(f, ESCALATE, CONTEST, ADVANTAGE).break_even_review_cost_inr
            for f in (0.0, 0.05, 0.2, 0.5)
        ]
        assert costs == sorted(costs, reverse=True)

    def test_no_advantage_means_no_break_even_to_erode(self):
        point = review_cost_point(0.1, ESCALATE, CONTEST, advantage_per_1000_inr=-50.0)
        assert point.break_even_review_cost_inr == 0.0

    def test_the_curve_covers_every_requested_rate(self):
        points = review_cost_curve(ESCALATE, CONTEST, ADVANTAGE, false_flag_rates=(0.0, 0.1))
        assert [p.false_flag_rate for p in points] == [0.0, 0.1]


class TestInverse:
    def test_the_inverse_agrees_with_the_forward_calculation(self):
        """Round trip: the rate that cancels the advantage at cost C, put back
        through the forward calculation, must break even at about C."""
        for cost in (60.0, 100.0, 150.0):
            flag = false_flag_rate_at_review_cost(cost, ESCALATE, CONTEST, ADVANTAGE)
            point = review_cost_point(flag, ESCALATE, CONTEST, ADVANTAGE)
            assert point.break_even_review_cost_inr == pytest.approx(cost, rel=1e-6)

    def test_the_repo_s_own_analyst_time_figure_leaves_a_narrow_budget(self):
        """`policy/config.py` budgets INR 150 of analyst time per contested
        dispute. At that price the gate's false-flag rate has almost no room:
        this pins the finding so a later change cannot quietly widen it."""
        flag = false_flag_rate_at_review_cost(150.0, ESCALATE, CONTEST, ADVANTAGE)
        assert flag == pytest.approx(0.023, abs=0.002)

    def test_at_the_zero_flag_break_even_no_room_is_left(self):
        assert false_flag_rate_at_review_cost(200.0, ESCALATE, CONTEST, ADVANTAGE) == 0.0

    def test_a_non_positive_advantage_has_no_answer(self):
        assert math.isnan(false_flag_rate_at_review_cost(150.0, ESCALATE, CONTEST, 0.0))
