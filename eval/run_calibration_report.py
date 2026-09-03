"""CLI entry point for the calibration diagnostic (Phase 2 addendum: the
unchecked Elkan precondition). The EV threshold `contest iff p_win >
cost/amount` is only optimal on a calibrated posterior; `eval.run_harness`
already reports ECE per seed, but never Brier score, a reliability table, or
calibration specifically in the region where a gap would flip a decision.
This script adds those, purely as a read on existing model predictions - it
does not train, retrain, or wrap the model in a calibrator.

Run as `python -m eval.run_calibration_report`.
"""

import argparse

import numpy as np
import pandas as pd

from disputedesk.generator.config import GeneratorConfig
from disputedesk.model.config import ModelConfig
from disputedesk.policy.config import PolicyConfig
from eval.calibration import (
    brier_score,
    calibration_table,
    expected_calibration_error,
    near_threshold_reliability,
)
from eval.harness import LABEL_COLUMN, fixed_seed_set, run_seed_pipeline


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Calibration diagnostic for the Elkan EV threshold's unchecked precondition."
    )
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-rows", type=int, default=15000)
    parser.add_argument("--band", type=float, default=0.05)
    args = parser.parse_args(argv)

    seeds = fixed_seed_set(args.n_seeds, start=args.seed_start)
    cost = PolicyConfig().representment_cost_inr

    per_seed_rows = []
    pooled_p, pooled_y, pooled_amount = [], [], []
    for seed in seeds:
        run = run_seed_pipeline(seed, args.n_rows, GeneratorConfig(), ModelConfig())
        y_test = run.test_df[LABEL_COLUMN].to_numpy()
        amount = run.test_df["amount"].to_numpy()
        per_seed_rows.append(
            {
                "seed": seed,
                "brier_score": brier_score(run.predicted_p, y_test),
                "calibration_error": expected_calibration_error(run.predicted_p, y_test),
            }
        )
        pooled_p.append(run.predicted_p)
        pooled_y.append(y_test)
        pooled_amount.append(amount)

    per_seed = pd.DataFrame(per_seed_rows)
    pooled_p = np.concatenate(pooled_p)
    pooled_y = np.concatenate(pooled_y)
    pooled_amount = np.concatenate(pooled_amount)

    print(f"seeds: {seeds[0]}..{seeds[-1]} (n={len(seeds)}), n_rows per seed: {args.n_rows}")
    print()
    print("Per-seed Brier / ECE, median+IQR (no single-seed headline - CLAUDE.md inv. 3):")
    for col in ("brier_score", "calibration_error"):
        s = per_seed[col]
        low, high = s.quantile(0.25), s.quantile(0.75)
        print(f"  {col:20s} median {s.median():.4f}  IQR {low:.4f}-{high:.4f}")
    print()

    print(f"Reliability table, pooled across all {len(seeds)} seeds' holdouts (n={len(pooled_p)}):")
    table = calibration_table(pooled_p, pooled_y, n_bins=10)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    result = near_threshold_reliability(pooled_p, pooled_y, pooled_amount, cost, band=args.band)
    print(
        f"Near-threshold reliability (rows where predicted p_win is within "
        f"+/-{args.band} of that row's own derived threshold cost/amount, "
        f"at the configured Rs{cost:.0f} cost, pooled across all seeds):"
    )
    median_threshold = result["median_threshold_overall"]
    print(f"  overall median derived threshold (cost/amount): {median_threshold:.4f}")
    print(f"  count in band:      {result['count']} of {len(pooled_p)}")
    print(f"  mean predicted p:   {result['mean_predicted_p']:.4f}")
    print(f"  observed win rate:  {result['observed_win_rate']:.4f}")
    print(f"  gap (pred - obs):   {result['gap']:.4f}")


if __name__ == "__main__":
    main()
