"""CLI entry point for the eval harness: runs the model across a fixed seed
set and prints/writes the median/IQR headline report (PHASES.md Phase 2 gate:
every headline number as median and IQR across at least 20 seeds).

Run as `python -m eval.run_harness`.
"""

import argparse
from pathlib import Path

from disputedesk.generator.config import GeneratorConfig
from disputedesk.model.config import ModelConfig
from eval.harness import fixed_seed_set, run_harness, summarize
from eval.report import format_precision_recall_headline


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 2 eval harness (GENERATOR.md).")
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-rows", type=int, default=15000)
    parser.add_argument("--out-dir", type=Path, default=Path("data/eval"))
    args = parser.parse_args(argv)

    seeds = fixed_seed_set(args.n_seeds, start=args.seed_start)
    per_seed = run_harness(seeds, args.n_rows, GeneratorConfig(), ModelConfig())
    headline = summarize(per_seed)
    precision_recall_headline = format_precision_recall_headline(headline)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Per-seed rows are not a headline number on their own (CLAUDE.md invariant
    # 3) - the filename says so, so nobody quotes row 0 as "the" result.
    per_seed.to_csv(args.out_dir / "per_seed_do_not_report_individually.csv", index=False)
    headline.to_csv(args.out_dir / "headline_median_iqr.csv")
    # The one file meant to be copied verbatim into the README / pitch video
    # for submission checklist items 3 and 4 - precision and recall never
    # appear here without the threshold they were measured at.
    (args.out_dir / "precision_recall_headline.txt").write_text(precision_recall_headline + "\n")

    print(f"seeds: {seeds[0]}..{seeds[-1]} (n={len(seeds)}), n_rows per seed: {args.n_rows}")
    print()
    print(headline.to_string(float_format=lambda x: f"{x:.4f}"))
    print()
    print(precision_recall_headline)
    print()
    print(f"wrote {args.out_dir / 'per_seed_do_not_report_individually.csv'}")
    print(f"wrote {args.out_dir / 'headline_median_iqr.csv'}")
    print(f"wrote {args.out_dir / 'precision_recall_headline.txt'}")


if __name__ == "__main__":
    main()
