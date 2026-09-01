"""CLI for the `representment_cost_inr` sensitivity sweep (reported alongside
the configured value, not a retune - see `eval/cost_sensitivity.py`).

Run as `python -m eval.run_cost_sensitivity`.
"""

import argparse
from pathlib import Path

import pandas as pd

from disputedesk.generator.config import GeneratorConfig
from disputedesk.model.config import ModelConfig
from disputedesk.policy.config import REPRESENTMENT_COST_INR
from eval.cost_sensitivity import fixed_seed_set, summarize_sweep, sweep_representment_cost


def _fmt_median_iqr(summary: pd.DataFrame, prefix: str) -> pd.Series:
    return (
        summary[f"{prefix}_median"].map("{:.4f}".format)
        + " (IQR "
        + summary[f"{prefix}_q25"].map("{:.4f}".format)
        + "-"
        + summary[f"{prefix}_q75"].map("{:.4f}".format)
        + ")"
    )


DEFAULT_COSTS = [
    0,
    50,
    100,
    200,
    300,
    400,
    600,
    800,
    1000,
    1500,
    2000,
    3000,
    4000,
    6000,
    8000,
    10000,
]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Sweep representment_cost_inr, holding the model/seeds/band fixed."
    )
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-rows", type=int, default=15000)
    parser.add_argument("--costs", type=float, nargs="+", default=DEFAULT_COSTS)
    parser.add_argument("--out-dir", type=Path, default=Path("data/eval"))
    args = parser.parse_args(argv)

    seeds = fixed_seed_set(args.n_seeds, start=args.seed_start)
    per_seed_cost = sweep_representment_cost(
        seeds, args.n_rows, args.costs, GeneratorConfig(), ModelConfig()
    )
    summary = summarize_sweep(per_seed_cost)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_seed_cost.to_csv(
        args.out_dir / "cost_sensitivity_per_seed_do_not_report_individually.csv", index=False
    )
    summary.to_csv(args.out_dir / "cost_sensitivity_median_iqr.csv", index=False)

    print(f"seeds: {seeds[0]}..{seeds[-1]} (n={len(seeds)}), n_rows per seed: {args.n_rows}")
    print(f"configured representment_cost_inr (unchanged): {REPRESENTMENT_COST_INR}")
    print(
        "escalate_mode: naive_contest (fair vs. baseline A - see eval.business_metrics); "
        "same mode used to fold ESCALATE into the positive prediction for "
        "precision/recall below (see eval.cost_sensitivity._predicted_positive)"
    )
    print()
    print(summary.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))
    print()

    table = pd.DataFrame(
        {
            "representment_cost_inr": summary["representment_cost_inr"],
            "precision": _fmt_median_iqr(summary, "precision"),
            "recall": _fmt_median_iqr(summary, "recall"),
            "escalate_rate": _fmt_median_iqr(summary, "escalate_rate"),
            "policy_advantage_recovered_per_1000_inr": summary["policy_advantage_median"].map(
                "{:,.1f}".format
            ),
        }
    )
    print(
        "cost, precision, recall, escalate rate, and policy's rupees-recovered "
        "advantage over baseline A:"
    )
    print(table.to_string(index=False))
    print()

    crossover = summary[summary["policy_advantage_median"] <= 0]
    if crossover.empty:
        print("policy_advantage_median > 0 at every swept cost - no crossover found in range.")
    else:
        last_negative_or_zero = crossover.iloc[-1]["representment_cost_inr"]
        print(
            "policy_advantage_median <= 0 up to and including cost = "
            f"{last_negative_or_zero:,.0f} within the swept range."
        )

    print()
    print(f"wrote {args.out_dir / 'cost_sensitivity_per_seed_do_not_report_individually.csv'}")
    print(f"wrote {args.out_dir / 'cost_sensitivity_median_iqr.csv'}")


if __name__ == "__main__":
    main()
