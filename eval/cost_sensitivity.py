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
"""

import numpy as np
import pandas as pd

from disputedesk.generator.config import GeneratorConfig
from disputedesk.model.config import ModelConfig
from disputedesk.policy.config import PolicyConfig
from eval.business_metrics import (
    contest_everything_recovered,
    decide_batch,
    per_1000,
    recovered_rupees,
)
from eval.harness import LABEL_COLUMN, run_seed_pipeline

ESCALATE_MODE = "naive_contest"


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
        rows.append(
            {
                "seed": seed,
                "representment_cost_inr": cost,
                "n": n,
                "policy_recovered_per_1000_inr": per_1000(policy_recovered.sum(), n),
                "baseline_a_recovered_per_1000_inr": per_1000(baseline_a_recovered.sum(), n),
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
    """Median and IQR of both recovered-rupee series, and their difference
    (the policy's advantage over baseline A), one row per cost value.
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
        )
        .reset_index()
        .sort_values("representment_cost_inr")
    )
    summary["policy_advantage_median"] = summary["policy_median"] - summary["baseline_a_median"]
    return summary


def fixed_seed_set(n_seeds: int, start: int = 0) -> list[int]:
    return list(np.arange(start, start + n_seeds))
