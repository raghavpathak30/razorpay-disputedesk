"""CLI entry point for the escalate-band counterfactual (Phase 2 addendum
item 2): what the ~5.6%-of-holdout escalate band costs or saves versus
letting those same disputes follow the plain EV rule instead. Eval-only -
`disputedesk/policy/engine.py` and `disputedesk/policy/config.py` are never
touched or called with a modified config.

Run as `python -m eval.run_escalate_band_counterfactual`.
"""

import argparse

from disputedesk.generator.config import GeneratorConfig
from disputedesk.model.config import ModelConfig
from disputedesk.policy.config import PolicyConfig
from eval.escalate_band_counterfactual import run_band_counterfactual, summarize_band_counterfactual
from eval.harness import fixed_seed_set
from eval.paired import format_paired


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Escalate-band counterfactual (Phase 2 addendum).")
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-rows", type=int, default=15000)
    args = parser.parse_args(argv)

    seeds = fixed_seed_set(args.n_seeds, start=args.seed_start)
    cost = PolicyConfig().representment_cost_inr
    band = PolicyConfig().low_confidence_band

    results = run_band_counterfactual(
        seeds, args.n_rows, cost, GeneratorConfig(), ModelConfig(), band
    )
    summary = summarize_band_counterfactual(results)

    print(f"seeds: {seeds[0]}..{seeds[-1]} (n={len(seeds)}), n_rows per seed: {args.n_rows}")
    print(f"configured representment_cost_inr: {cost}, low_confidence_band: {band}")
    print()
    print(
        f"escalate rate: median {summary['escalate_rate_median']:.4f} "
        f"(IQR {summary['escalate_rate_q25']:.4f}-{summary['escalate_rate_q75']:.4f})"
    )
    print()
    print("Actual (banded) policy advantage vs. baseline A:")
    print(f"  {format_paired(summary['actual_advantage'], unit='INR/1,000')}")
    print()
    print("Counterfactual (band-free EV rule) advantage vs. baseline A:")
    print(f"  {format_paired(summary['counterfactual_advantage'], unit='INR/1,000')}")
    print()
    print(f"Delta (counterfactual - actual), paired mean: {summary['delta_mean']:,.1f} INR/1,000")


if __name__ == "__main__":
    main()
