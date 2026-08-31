"""CI-speed regression test for the cost-weighted business harness (SPEC.md
§6, PHASES.md Phase 3), mirroring `test_eval_harness_regression.py`'s
pattern: small n_rows/seed count for CI speed, structural/sanity bounds
rather than the headline numbers (produced by `eval/run_business_harness.py`
at n_rows=15000, >=20 seeds).
"""

import pytest

from disputedesk.generator.config import GeneratorConfig
from disputedesk.model.config import ModelConfig
from disputedesk.policy.config import PolicyConfig
from eval.business_harness import fixed_seed_set, policy_beats_baseline, run_business_harness
from eval.business_metrics import summarize_business

CI_SEEDS = fixed_seed_set(8, start=0)
CI_N_ROWS = 5000


@pytest.fixture(scope="module")
def ci_business_summary():
    per_seed = run_business_harness(
        CI_SEEDS, CI_N_ROWS, GeneratorConfig(), ModelConfig(), PolicyConfig()
    )
    return per_seed, summarize_business(per_seed)


def test_business_harness_runs_end_to_end_on_the_fixed_ci_seed_set(ci_business_summary):
    per_seed, _summary = ci_business_summary
    assert len(per_seed) == len(CI_SEEDS)
    assert set(per_seed["seed"]) == set(CI_SEEDS)


def test_policy_beats_accept_everything_on_median_recovered(ci_business_summary):
    # Accepting everything recovers INR 0 by construction (the reference
    # outcome) - a policy that ever nets negative here would mean contesting
    # is actively destroying value relative to doing nothing, which would
    # signal a broken policy/cost wiring, not just an unfavorable baseline
    # comparison.
    _per_seed, summary = ci_business_summary
    beats = policy_beats_baseline(summary)
    assert beats["beats_baseline_b_accept_everything"] is True


def test_false_positive_and_false_negative_counts_are_non_negative(ci_business_summary):
    per_seed, _summary = ci_business_summary
    assert (per_seed["false_positive_count"] >= 0).all()
    assert (per_seed["false_negative_count"] >= 0).all()
    assert (per_seed["escalated_count"] >= 0).all()


def test_false_positive_cost_matches_the_fixed_representment_cost_times_count(ci_business_summary):
    per_seed, _summary = ci_business_summary
    cost = PolicyConfig().representment_cost_inr
    implied_total = per_seed["false_positive_cost_per_1000_inr"] * per_seed["n"] / 1000.0
    expected_total = per_seed["false_positive_count"] * cost
    assert (implied_total.round(2) == expected_total.round(2)).all()


def test_escalated_and_decided_disputes_do_not_exceed_the_test_split_size(ci_business_summary):
    per_seed, _summary = ci_business_summary
    # Every dispute lands in exactly one of contest/accept/escalate, so
    # (n - escalated) must be >= 0 and escalated must never exceed n.
    assert (per_seed["escalated_count"] <= per_seed["n"]).all()
