"""CLI entry point for the cost-weighted eval harness (SPEC.md §6, PHASES.md
Phase 3): runs the policy engine across a fixed seed set and prints/writes
the median/IQR rupees-recovered and FP/FN cost report.

Run as `python -m eval.run_business_harness`.
"""

import argparse
from pathlib import Path

from disputedesk.generator.config import GeneratorConfig
from disputedesk.model.config import ModelConfig
from disputedesk.policy.config import PolicyConfig
from eval.business_harness import fixed_seed_set, policy_beats_baseline, run_business_harness
from eval.business_metrics import summarize_business


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the Phase 3 cost-weighted eval harness (SPEC.md §6)."
    )
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-rows", type=int, default=15000)
    parser.add_argument("--out-dir", type=Path, default=Path("data/eval"))
    args = parser.parse_args(argv)

    seeds = fixed_seed_set(args.n_seeds, start=args.seed_start)
    per_seed = run_business_harness(
        seeds, args.n_rows, GeneratorConfig(), ModelConfig(), PolicyConfig()
    )
    summary = summarize_business(per_seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(args.out_dir / "business_per_seed_do_not_report_individually.csv", index=False)
    summary.to_csv(args.out_dir / "business_headline_median_iqr.csv")

    print(f"seeds: {seeds[0]}..{seeds[-1]} (n={len(seeds)}), n_rows per seed: {args.n_rows}")
    print(f"representment_cost_inr: {PolicyConfig().representment_cost_inr}")
    print()
    print(summary.to_string(float_format=lambda x: f"{x:.2f}"))
    print()
    print(
        "escalated_amount_share_of_holdout (median): "
        f"{summary.loc['escalated_amount_share_of_holdout', 'median']:.4f} "
        "of total holdout rupees sit in escalated disputes"
    )
    print()
    print("beats each baseline, by what an escalated dispute is credited:")
    for escalate_mode in ("zero", "oracle", "naive_contest"):
        beats = policy_beats_baseline(summary, escalate_mode=escalate_mode)
        print(f"  escalate_mode={escalate_mode}:")
        for label, result in beats.items():
            print(f"    {label}: {'YES' if result else 'NO'}")

    naive_contest_beats = policy_beats_baseline(summary, escalate_mode="naive_contest")
    if not naive_contest_beats["beats_baseline_a_contest_everything"]:
        print()
        print(
            "The policy does not beat contest-everything on median recovered "
            "rupees per 1,000 disputes even under escalate_mode=naive_contest "
            "(escalated disputes scored exactly as baseline A already scores "
            "them - the fairest comparison). Reported as the result, per "
            "CLAUDE.md/PHASES.md - not adjusted to make it win."
        )
    print()
    print(f"wrote {args.out_dir / 'business_per_seed_do_not_report_individually.csv'}")
    print(f"wrote {args.out_dir / 'business_headline_median_iqr.csv'}")


if __name__ == "__main__":
    main()
