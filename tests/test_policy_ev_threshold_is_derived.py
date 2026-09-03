"""Pins Phase 2 STEP A's finding (DECISIONS.md 2026-09-03 "EV threshold
already derived"): the policy engine's CONTEST/ACCEPT decision already IS
the Elkan (2001) cost-sensitive threshold p* = contest_cost / amount,
per-dispute — not a hand-chosen scalar constant like p >= 0.5. `decide()`
was never refactored for this; SPEC.md §4 has specified
`expected_value = P(win) * amount - representment_cost` since Phase 3, and
that is algebraically `p > cost / amount` whenever `amount > 0`. This test
evaluates the equivalence on real holdout predictions, not just synthetic
point cases (`tests/test_policy_engine.py` already covers those), so a
future change to `decide()` that quietly breaks the equivalence fails here.
"""

import numpy as np

from disputedesk.generator.config import GeneratorConfig
from disputedesk.model.config import ModelConfig
from disputedesk.policy.config import PolicyConfig
from disputedesk.policy.engine import Decision, decide
from eval.harness import run_seed_pipeline

CI_SEED = 0
CI_N_ROWS = 5000
CONFIG = PolicyConfig()  # representment_cost_inr=400.0, band=(0.45, 0.55) — the configured values


def _holdout_predictions_and_amounts() -> tuple[np.ndarray, np.ndarray]:
    run = run_seed_pipeline(CI_SEED, CI_N_ROWS, GeneratorConfig(), ModelConfig())
    return run.predicted_p, run.test_df["amount"].to_numpy()


def test_contest_decision_equals_the_per_dispute_ev_threshold_on_the_holdout():
    """Outside the escalate band, `decide()`'s CONTEST branch and
    `p_win > representment_cost / amount` must agree on every holdout row,
    at the configured Rs400 cost. This is what "already derived from the
    cost model" means, not an approximation.
    """
    predicted_p, amount = _holdout_predictions_and_amounts()
    low, high = CONFIG.low_confidence_band

    checked = 0
    for p, a in zip(predicted_p, amount, strict=True):
        p, a = float(p), float(a)
        if low <= p <= high:
            continue  # the escalate band overrides the EV comparison entirely
        result = decide(p, a, config=CONFIG)
        derived_contest = p > (CONFIG.representment_cost_inr / a)
        assert (result.decision == Decision.CONTEST) == derived_contest
        checked += 1

    assert checked > 0  # the band shouldn't swallow the whole holdout


def test_the_derived_threshold_varies_per_dispute_not_a_single_scalar():
    """p* = cost / amount is not one number to swap in for another — it
    tracks `amount`, which varies row to row on the holdout.
    """
    _predicted_p, amount = _holdout_predictions_and_amounts()
    thresholds = CONFIG.representment_cost_inr / amount
    assert len(np.unique(np.round(thresholds, 6))) > 1
