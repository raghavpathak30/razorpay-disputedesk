"""Cost-weighted evaluation, deferred from Phase 2 to Phase 3 by PHASES.md
(the policy engine's `representment_cost` didn't exist yet). One full
generate -> split -> train -> predict -> policy-decide cycle per seed,
reusing `eval.harness.run_seed_pipeline` so the rupee metrics are computed on
exactly the same holdout predictions the model-quality metrics use.

Reported on the temporal holdout only (CLAUDE.md invariant 2), median and IQR
across seeds (invariant 3, PHASES.md Phase 2 gate: >=20 seeds).
"""

import numpy as np
import pandas as pd

from disputedesk.generator.config import GeneratorConfig
from disputedesk.model.config import ModelConfig
from disputedesk.policy.config import PolicyConfig
from eval.business_metrics import build_business_row
from eval.harness import LABEL_COLUMN, run_seed_pipeline


def run_business_seed(
    seed: int,
    n_rows: int,
    generator_config: GeneratorConfig,
    model_config: ModelConfig,
    policy_config: PolicyConfig,
) -> dict:
    run = run_seed_pipeline(seed, n_rows, generator_config, model_config)
    amount = run.test_df["amount"].to_numpy()
    won_if_contested = run.test_df[LABEL_COLUMN].to_numpy()

    row = build_business_row(run.predicted_p, won_if_contested, amount, policy_config)
    return {"seed": seed, **row}


def run_business_harness(
    seeds: list[int],
    n_rows: int,
    generator_config: GeneratorConfig | None = None,
    model_config: ModelConfig | None = None,
    policy_config: PolicyConfig | None = None,
) -> pd.DataFrame:
    """Run `run_business_seed` across every seed and return one row per seed."""
    generator_config = generator_config or GeneratorConfig()
    model_config = model_config or ModelConfig()
    policy_config = policy_config or PolicyConfig()

    rows = [
        run_business_seed(seed, n_rows, generator_config, model_config, policy_config)
        for seed in seeds
    ]
    return pd.DataFrame(rows)


_POLICY_METRIC_BY_ESCALATE_MODE = {
    "zero": "policy_recovered_per_1000_inr",
    "oracle": "policy_recovered_per_1000_inr_escalate_oracle",
    "naive_contest": "policy_recovered_per_1000_inr_escalate_naive_contest",
}


def policy_beats_baseline(summary: pd.DataFrame, escalate_mode: str = "zero") -> dict[str, bool]:
    """Whether the policy's median recovered-per-1,000 beats each baseline's
    median, under one `escalate_mode` (see `eval.business_metrics`'s module
    docstring for what each mode credits an escalated dispute).
    CLAUDE.md invariant/PHASES.md gate: if it does not beat both, that is
    reported as the result, not adjusted away.
    """
    policy_median = summary.loc[_POLICY_METRIC_BY_ESCALATE_MODE[escalate_mode], "median"]
    return {
        "beats_baseline_a_contest_everything": bool(
            policy_median
            > summary.loc["baseline_a_contest_everything_recovered_per_1000_inr", "median"]
        ),
        "beats_baseline_b_accept_everything": bool(
            policy_median
            > summary.loc["baseline_b_accept_everything_recovered_per_1000_inr", "median"]
        ),
    }


def fixed_seed_set(n_seeds: int, start: int = 0) -> list[int]:
    """Same convention as `eval.harness.fixed_seed_set` - the reproducible
    seed list headline business numbers are reported against.
    """
    return list(np.arange(start, start + n_seeds))
