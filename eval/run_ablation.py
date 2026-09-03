"""CLI for the single-feature ablation (Phase 2 addendum item B): the
business-harness cost sweep run three times, restricting the model to the
top feature, the top three features, and the full set - same seeds, same
paired estimator as `eval.run_cost_sensitivity`.

Run as `python -m eval.run_ablation`.
"""

import argparse
from pathlib import Path

import pandas as pd

from disputedesk.features.build import FEATURE_COLUMNS
from disputedesk.generator.config import GeneratorConfig
from disputedesk.model.config import ModelConfig
from eval.ablation import TOP_1_FEATURE, TOP_3_FEATURES, sweep_feature_subset
from eval.cost_sensitivity import summarize_sweep
from eval.harness import fixed_seed_set

VARIANTS: dict[str, tuple[str, ...]] = {
    "top1": (TOP_1_FEATURE,),
    "top3": TOP_3_FEATURES,
    "full": FEATURE_COLUMNS,
}

DEFAULT_COSTS = [0, 50, 100, 200, 300, 400, 600, 800, 1000, 2000, 4000, 6000, 10000]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Single-feature ablation of the business-harness cost sweep."
    )
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-rows", type=int, default=15000)
    parser.add_argument("--costs", type=float, nargs="+", default=DEFAULT_COSTS)
    parser.add_argument("--out-dir", type=Path, default=Path("data/eval"))
    args = parser.parse_args(argv)

    seeds = fixed_seed_set(args.n_seeds, start=args.seed_start)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"seeds: {seeds[0]}..{seeds[-1]} (n={len(seeds)}), n_rows per seed: {args.n_rows}")
    print(f"top-1 feature: {TOP_1_FEATURE}")
    print(f"top-3 features: {TOP_3_FEATURES}")
    print(f"full feature set ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")
    print()

    summaries = {}
    for name, feature_columns in VARIANTS.items():
        per_seed = sweep_feature_subset(
            seeds, args.n_rows, args.costs, feature_columns, GeneratorConfig(), ModelConfig()
        )
        summary = summarize_sweep(per_seed)
        summaries[name] = summary
        per_seed.to_csv(
            args.out_dir / f"ablation_{name}_per_seed_do_not_report_individually.csv",
            index=False,
        )
        summary.to_csv(args.out_dir / f"ablation_{name}_median_iqr.csv", index=False)
        print(f"wrote {args.out_dir}/ablation_{name}_*.csv")

    print()
    print("advantage vs. baseline A (paired mean, 95% CI, seeds positive), by variant and cost:")
    rows = []
    for cost in args.costs:
        row = {"cost": cost}
        for name in VARIANTS:
            s = summaries[name].set_index("representment_cost_inr").loc[cost]
            row[f"{name}_mean"] = s["advantage_paired_mean"]
            row[f"{name}_ci"] = f"{s['advantage_ci_low']:,.0f} to {s['advantage_ci_high']:,.0f}"
            row[f"{name}_pos"] = f"{int(s['advantage_n_positive'])}/{int(s['n_seeds'])}"
        rows.append(row)
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))


if __name__ == "__main__":
    main()
