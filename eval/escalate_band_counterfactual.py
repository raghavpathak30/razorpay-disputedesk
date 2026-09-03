"""What-if: what would the policy's advantage over baseline A be if
`low_confidence_band` did not exist, and every dispute simply followed the
EV rule (`contest iff p_win * amount > representment_cost`) instead of the
~5.6% of holdout rows the band sends to `Decision.ESCALATE`?

This is eval-only, by design (Phase 2 addendum item 2): `decide()`,
`PolicyConfig`, and `disputedesk/policy/engine.py` are never touched or
called with a modified band. The counterfactual decision rule is
reimplemented here, read-only, purely to quantify what the band already
costs or saves - not to change what the running system does.

`decide()`'s own low-confidence check runs *before* the cost-dependent
branch and never reads `cost` or `amount`, so it is a real behavioral
departure from EV-optimality, not an approximation of it: it exists to buy
human review of the genuinely uncertain region (SPEC.md §4), and this module
puts a number on what that trade costs in this sweep's terms.
"""

import numpy as np
import pandas as pd

from disputedesk.generator.config import GeneratorConfig
from disputedesk.model.config import ModelConfig
from disputedesk.policy.config import PolicyConfig
from disputedesk.policy.engine import Decision
from eval.business_metrics import (
    contest_everything_recovered,
    decide_batch,
    per_1000,
    recovered_rupees,
)
from eval.harness import LABEL_COLUMN, run_seed_pipeline
from eval.paired import paired_difference


def no_band_decisions(p_win: np.ndarray, amount: np.ndarray, cost: float) -> np.ndarray:
    """The plain Elkan EV rule, applied to every row with no escalate carve-out
    at all: `contest iff p_win * amount - cost > 0`. Mirrors `decide()`'s own
    `expected_value` arithmetic exactly (`disputedesk/policy/engine.py`) minus
    the low-confidence-band branch - reimplemented here rather than imported,
    so this stays a read-only what-if with zero coupling to policy internals
    that could change out from under it silently.
    """
    p_win = np.asarray(p_win, dtype=float)
    amount = np.asarray(amount, dtype=float)
    expected_value = p_win * amount - cost
    return np.where(expected_value > 0.0, Decision.CONTEST, Decision.ACCEPT)


def score_seed(
    seed: int,
    n_rows: int,
    generator_config: GeneratorConfig,
    model_config: ModelConfig,
    cost: float,
    low_confidence_band: tuple[float, float],
) -> dict:
    """One seed: the actual (banded) policy's recovered rupees, the
    band-free-EV counterfactual's recovered rupees, and baseline A's -
    all against the same holdout and the same trained model's predictions.
    """
    run = run_seed_pipeline(seed, n_rows, generator_config, model_config)
    amount = run.test_df["amount"].to_numpy()
    won_if_contested = run.test_df[LABEL_COLUMN].to_numpy()
    p_win = run.predicted_p
    n = len(amount)

    config = PolicyConfig(representment_cost_inr=cost, low_confidence_band=low_confidence_band)
    actual_decisions = decide_batch(p_win, amount, config)
    escalate_rate = float(np.mean(actual_decisions == Decision.ESCALATE))
    # naive_contest: the same convention the published +11,210.3 headline
    # uses (eval.cost_sensitivity.ESCALATE_MODE) - an escalated row is
    # credited exactly as baseline A already treats it, for a fair
    # apples-to-apples comparison. Isolates the effect of the band itself.
    actual_recovered = recovered_rupees(
        actual_decisions, won_if_contested, amount, cost, escalate_mode="naive_contest"
    )

    counterfactual_decisions = no_band_decisions(p_win, amount, cost)
    counterfactual_recovered = recovered_rupees(
        counterfactual_decisions, won_if_contested, amount, cost, escalate_mode="naive_contest"
    )

    baseline_a_recovered = contest_everything_recovered(won_if_contested, amount, cost)

    return {
        "seed": seed,
        "n": n,
        "escalate_rate": escalate_rate,
        "actual_policy_recovered_per_1000_inr": per_1000(actual_recovered.sum(), n),
        "counterfactual_policy_recovered_per_1000_inr": per_1000(counterfactual_recovered.sum(), n),
        "baseline_a_recovered_per_1000_inr": per_1000(baseline_a_recovered.sum(), n),
    }


def run_band_counterfactual(
    seeds: list[int],
    n_rows: int,
    cost: float,
    generator_config: GeneratorConfig | None = None,
    model_config: ModelConfig | None = None,
    low_confidence_band: tuple[float, float] = (0.45, 0.55),
) -> pd.DataFrame:
    """One row per seed. `low_confidence_band` defaults to `PolicyConfig`'s
    own default, matching the configured system exactly."""
    generator_config = generator_config or GeneratorConfig()
    model_config = model_config or ModelConfig()
    rows = [
        score_seed(seed, n_rows, generator_config, model_config, cost, low_confidence_band)
        for seed in seeds
    ]
    return pd.DataFrame(rows)


def summarize_band_counterfactual(results: pd.DataFrame, random_state: int = 0) -> dict:
    """Paired advantage over baseline A for both the actual (banded) policy
    and the band-free-EV counterfactual, plus the delta between them - the
    cost (or saving) of running the escalate band instead of the plain EV
    rule, in the same paired-mean terms the published headline uses.
    """
    ordered = results.sort_values("seed")
    baseline_a = ordered["baseline_a_recovered_per_1000_inr"].to_numpy()

    actual_advantage = paired_difference(
        ordered["actual_policy_recovered_per_1000_inr"].to_numpy(),
        baseline_a,
        random_state=random_state,
    )
    counterfactual_advantage = paired_difference(
        ordered["counterfactual_policy_recovered_per_1000_inr"].to_numpy(),
        baseline_a,
        random_state=random_state,
    )

    return {
        "escalate_rate_median": float(ordered["escalate_rate"].median()),
        "escalate_rate_q25": float(ordered["escalate_rate"].quantile(0.25)),
        "escalate_rate_q75": float(ordered["escalate_rate"].quantile(0.75)),
        "actual_advantage": actual_advantage,
        "counterfactual_advantage": counterfactual_advantage,
        "delta_mean": counterfactual_advantage.mean_difference - actual_advantage.mean_difference,
    }
