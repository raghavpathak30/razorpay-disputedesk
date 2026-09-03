"""Tests for the escalate-band counterfactual (Phase 2 addendum item 2):
what the escalate band costs or saves versus the plain EV rule, computed
eval-only with no change to `disputedesk/policy/`.
"""

import numpy as np
import pytest

from disputedesk.policy.engine import Decision, decide
from eval.escalate_band_counterfactual import (
    no_band_decisions,
    run_band_counterfactual,
    summarize_band_counterfactual,
)

CI_SEEDS = [0, 1, 2]
CI_N_ROWS = 3000
CI_COST = 400.0


def test_no_band_decisions_never_returns_escalate():
    p_win = np.array([0.1, 0.5, 0.9])
    amount = np.array([1000.0, 1000.0, 1000.0])
    decisions = no_band_decisions(p_win, amount, cost=400.0)
    assert set(decisions) <= {Decision.CONTEST, Decision.ACCEPT}


def test_no_band_decisions_matches_decide_ev_sign_outside_the_band():
    # decide()'s own low_confidence_band is (0.45, 0.55) by default; pick
    # p values clear of it so decide() and the band-free rule must agree.
    p_win = np.array([0.05, 0.2, 0.8, 0.95])
    amount = np.array([2000.0, 3000.0, 500.0, 100.0])
    cost = 400.0

    band_free = no_band_decisions(p_win, amount, cost)
    for i, (p, a) in enumerate(zip(p_win, amount, strict=True)):
        result = decide(float(p), float(a), config=None)
        # default PolicyConfig has representment_cost_inr=400, matching `cost`
        assert band_free[i] == (
            Decision.CONTEST if result.decision == Decision.CONTEST else Decision.ACCEPT
        )
        assert result.decision != Decision.ESCALATE  # sanity: these p values are outside the band


def test_run_band_counterfactual_runs_end_to_end_on_a_small_seed_set():
    results = run_band_counterfactual(CI_SEEDS, CI_N_ROWS, CI_COST)
    assert len(results) == len(CI_SEEDS)
    assert (results["escalate_rate"] > 0).all()  # the band should bind on some rows at this scale


def test_band_cost_mean_matches_the_difference_of_the_two_advantage_means():
    # baseline_a(s) is identical in both advantage(s) terms for a given seed,
    # so it must cancel algebraically: band_cost's mean equals the difference
    # of the two separately-computed advantage means, even though its own CI
    # is a directly-paired bootstrap over (counterfactual - actual), not a
    # combination of the other two arms' marginal CIs.
    results = run_band_counterfactual(CI_SEEDS, CI_N_ROWS, CI_COST)
    summary = summarize_band_counterfactual(results)
    counterfactual = summary["counterfactual_advantage"].mean_difference
    actual = summary["actual_advantage"].mean_difference
    assert summary["band_cost"].mean_difference == pytest.approx(counterfactual - actual)


def test_band_cost_has_its_own_directly_paired_confidence_interval():
    results = run_band_counterfactual(CI_SEEDS, CI_N_ROWS, CI_COST)
    summary = summarize_band_counterfactual(results)
    band_cost = summary["band_cost"]
    assert band_cost.n_pairs == len(CI_SEEDS)
    assert band_cost.ci_low <= band_cost.mean_difference <= band_cost.ci_high
