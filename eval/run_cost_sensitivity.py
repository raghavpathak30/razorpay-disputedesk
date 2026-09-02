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


# 150/250/350 were added on 2026-09-02 to resolve exactly where the paired
# advantage becomes measurable - the old summary asserted "indistinguishable
# below ~290" from a grid too coarse to locate the transition.
DEFAULT_COSTS = [
    0,
    50,
    100,
    150,
    200,
    250,
    300,
    350,
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


def _print_advantage_verdict(summary: pd.DataFrame) -> None:
    """Where the paired advantage is and is not distinguishable from zero.

    Replaces the old "policy_advantage_median <= 0" crossover line, which
    tested a point estimate against zero with no interval and so could not
    distinguish "no advantage" from "advantage not measurable here".
    """
    indistinguishable = summary[~summary["advantage_excludes_zero"].astype(bool)]
    if indistinguishable.empty:
        print(
            "The 95% CI excludes zero at every swept cost - the paired advantage "
            "is measurable across the whole range."
        )
        return

    costs = ", ".join(f"{c:,.0f}" for c in indistinguishable["representment_cost_inr"])
    print(
        "No measurable advantage (95% CI includes zero) at representment cost: "
        f"{costs}. The CI excludes zero at every other swept cost."
    )
    negative = indistinguishable[indistinguishable["advantage_paired_mean"] < 0]
    if not negative.empty:
        neg_costs = ", ".join(f"{c:,.0f}" for c in negative["representment_cost_inr"])
        print(f"  Of those, the point estimate is negative at: {neg_costs}.")


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
            "paired_advantage_per_1000_inr": summary["advantage_paired_mean"].map(
                "{:,.1f}".format
            ),
            "ci_95": (
                summary["advantage_ci_low"].map("{:,.1f}".format)
                + " to "
                + summary["advantage_ci_high"].map("{:,.1f}".format)
            ),
            "seeds_positive": (
                summary["advantage_n_positive"].astype(int).astype(str)
                + "/"
                + summary["n_seeds"].astype(int).astype(str)
            ),
            "break_even_review_cost_inr": summary["break_even_human_review_cost_inr"].map(
                "{:,.0f}".format
            ),
        }
    )
    print(
        "cost, precision, recall, escalate rate, and the PAIRED advantage over "
        "baseline A (mean of per-seed differences, 95% bootstrap CI over seeds, "
        "sign-test count). `break_even_review_cost_inr` is the per-review cost "
        "at which that advantage is cancelled by human time this sweep does not "
        "charge for - computed at the ESCALATE rate alone, so it is an upper "
        "bound (see eval.cost_sensitivity's assumption 1):"
    )
    print(table.to_string(index=False))
    print()

    _print_advantage_verdict(summary)

    print()
    print(f"wrote {args.out_dir / 'cost_sensitivity_per_seed_do_not_report_individually.csv'}")
    print(f"wrote {args.out_dir / 'cost_sensitivity_median_iqr.csv'}")


if __name__ == "__main__":
    main()
