"""Sensitivity of the cost-weighted business metrics to
`representment_cost_inr`, holding the model, seeds, and `low_confidence_band`
fixed. This is a sensitivity analysis reported *alongside* the configured
value in `disputedesk.policy.config`, not a retune - nothing here writes to
that module, and the sweep never changes what the running system actually
uses.

`P(win)` does not depend on `representment_cost_inr` at all - only
`decide()` does - so each seed's generate/train/predict cycle
(`eval.harness.run_seed_pipeline`) runs exactly once per seed, and the sweep
over cost values reuses those same predictions. Escalated disputes are
scored under `escalate_mode="naive_contest"` throughout (see
`eval.business_metrics`'s module docstring and the 2026-08-31 "cost-weighted
business metrics" DECISIONS.md entry): that is the fair apples-to-apples
mode against baseline A, which already contests every escalated dispute.

Precision/recall of the policy's own decisions (CONTEST as the positive
prediction, `won_if_contested` as the label) are reported alongside the
recovered-rupee numbers at every swept cost. ESCALATE rows must be folded
into one side of that binary choice, and `_predicted_positive` below is
where that happens - kept in lockstep with `ESCALATE_MODE` on purpose, so
the same definition of "did the policy effectively contest this row" feeds
both the rupee and the precision/recall numbers. With `ESCALATE_MODE =
"naive_contest"`, escalated rows are credited as contested for recovery, so
they count as positive predictions here too.

The ESCALATE rate (fraction of holdout rows `decide()` sends to
`Decision.ESCALATE`) is also reported per swept cost, for the same reason
precision/recall is: it's a property of the decisions actually made at that
cost, not a fixed constant to take on faith. Structurally it should come out
*invariant* to `representment_cost_inr` - `decide()`'s low-confidence check
(`low <= p_win <= high`) runs before the cost-dependent
`expected_value`/CONTEST-vs-ACCEPT branch and never reads `cost` or `amount`,
so which rows escalate is fixed by `p_win` and `low_confidence_band` alone,
both held fixed across this sweep. Reported per cost anyway, not hardcoded
once, so that invariance is a measured fact, not an assumption.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

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

ESCALATE_MODE = "naive_contest"


def _predicted_positive(decisions: np.ndarray) -> np.ndarray:
    """CONTEST is always a positive prediction; ESCALATE's treatment must
    match `ESCALATE_MODE`'s treatment in `recovered_rupees` above so the
    precision/recall table and the rupee table describe the same decisions.
    Raises rather than guessing if `ESCALATE_MODE` is ever changed to a mode
    this function has not been updated for.
    """
    if ESCALATE_MODE != "naive_contest":
        raise NotImplementedError(
            f"no precision/recall mapping defined for ESCALATE_MODE={ESCALATE_MODE!r}"
        )
    return (decisions == Decision.CONTEST) | (decisions == Decision.ESCALATE)


def sweep_seed(
    seed: int,
    n_rows: int,
    costs: list[float],
    generator_config: GeneratorConfig,
    model_config: ModelConfig,
    low_confidence_band: tuple[float, float],
) -> list[dict]:
    """One seed's train/predict cycle, then every cost in `costs` scored
    against those same predictions - no retraining inside the loop.
    """
    run = run_seed_pipeline(seed, n_rows, generator_config, model_config)
    amount = run.test_df["amount"].to_numpy()
    won_if_contested = run.test_df[LABEL_COLUMN].to_numpy()
    n = len(amount)

    rows = []
    for cost in costs:
        config = PolicyConfig(representment_cost_inr=cost, low_confidence_band=low_confidence_band)
        decisions = decide_batch(run.predicted_p, amount, config)
        policy_recovered = recovered_rupees(
            decisions, won_if_contested, amount, cost, escalate_mode=ESCALATE_MODE
        )
        baseline_a_recovered = contest_everything_recovered(won_if_contested, amount, cost)
        predicted_positive = _predicted_positive(decisions)
        escalate_rate = float(np.mean(decisions == Decision.ESCALATE))
        rows.append(
            {
                "seed": seed,
                "representment_cost_inr": cost,
                "n": n,
                "policy_recovered_per_1000_inr": per_1000(policy_recovered.sum(), n),
                "baseline_a_recovered_per_1000_inr": per_1000(baseline_a_recovered.sum(), n),
                "policy_precision": precision_score(
                    won_if_contested, predicted_positive, zero_division=0
                ),
                "policy_recall": recall_score(
                    won_if_contested, predicted_positive, zero_division=0
                ),
                "policy_escalate_rate": escalate_rate,
            }
        )
    return rows


def sweep_representment_cost(
    seeds: list[int],
    n_rows: int,
    costs: list[float],
    generator_config: GeneratorConfig | None = None,
    model_config: ModelConfig | None = None,
    low_confidence_band: tuple[float, float] = (0.45, 0.55),
) -> pd.DataFrame:
    """One row per (seed, cost). `low_confidence_band` defaults to
    `PolicyConfig`'s own default, held fixed across the sweep per the "hold
    everything else fixed" instruction.
    """
    generator_config = generator_config or GeneratorConfig()
    model_config = model_config or ModelConfig()

    rows: list[dict] = []
    for seed in seeds:
        rows.extend(
            sweep_seed(seed, n_rows, costs, generator_config, model_config, low_confidence_band)
        )
    return pd.DataFrame(rows)


def summarize_sweep(results: pd.DataFrame) -> pd.DataFrame:
    """Median and IQR of both recovered-rupee series, the policy's own
    precision/recall (ESCALATE folded in per `_predicted_positive`), the
    ESCALATE rate itself, and the policy's advantage over baseline A, one row
    per cost value.
    """
    summary = (
        results.groupby("representment_cost_inr")
        .agg(
            policy_median=("policy_recovered_per_1000_inr", "median"),
            policy_q25=("policy_recovered_per_1000_inr", lambda s: s.quantile(0.25)),
            policy_q75=("policy_recovered_per_1000_inr", lambda s: s.quantile(0.75)),
            baseline_a_median=("baseline_a_recovered_per_1000_inr", "median"),
            baseline_a_q25=("baseline_a_recovered_per_1000_inr", lambda s: s.quantile(0.25)),
            baseline_a_q75=("baseline_a_recovered_per_1000_inr", lambda s: s.quantile(0.75)),
            precision_median=("policy_precision", "median"),
            precision_q25=("policy_precision", lambda s: s.quantile(0.25)),
            precision_q75=("policy_precision", lambda s: s.quantile(0.75)),
            recall_median=("policy_recall", "median"),
            recall_q25=("policy_recall", lambda s: s.quantile(0.25)),
            recall_q75=("policy_recall", lambda s: s.quantile(0.75)),
            escalate_rate_median=("policy_escalate_rate", "median"),
            escalate_rate_q25=("policy_escalate_rate", lambda s: s.quantile(0.25)),
            escalate_rate_q75=("policy_escalate_rate", lambda s: s.quantile(0.75)),
        )
        .reset_index()
        .sort_values("representment_cost_inr")
    )
    summary["policy_advantage_median"] = summary["policy_median"] - summary["baseline_a_median"]
    return summary


def fixed_seed_set(n_seeds: int, start: int = 0) -> list[int]:
    return list(np.arange(start, start + n_seeds))
