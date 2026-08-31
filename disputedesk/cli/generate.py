"""CLI entry point: generate a synthetic dispute dataset to disk from a seed and a
config, and print a short summary. Run as `python -m disputedesk.cli.generate`.
"""

import argparse
from pathlib import Path

from disputedesk.generator.config import GeneratorConfig
from disputedesk.generator.pipeline import generate_dataset, temporal_split


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic dispute dataset (GENERATOR.md)."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-rows", type=int, default=15000)
    parser.add_argument("--out-dir", type=Path, default=Path("data/generated"))
    args = parser.parse_args(argv)

    config = GeneratorConfig()
    features_df, debug_df = generate_dataset(args.n_rows, args.seed, config)
    train_df, test_df, boundary = temporal_split(features_df, config)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(args.out_dir / "disputes.csv", index=False)
    debug_df.to_csv(args.out_dir / "debug.csv", index=False)

    e_p = debug_df["p"].mean()
    fraud_rate = debug_df["true_fraud"].mean()
    component_shares = debug_df["component"].value_counts(normalize=True).sort_index()

    print(f"seed: {args.seed}")
    print(f"rows generated: {len(features_df)}")
    print(f"realised E[p]: {e_p:.4f} (target {config.e_p_target})")
    print(f"realised true_fraud rate: {fraud_rate:.4f}")
    print("confounder / component shares:")
    for name, share in component_shares.items():
        print(f"  {name}: {share:.4f}")
    print(f"train rows: {len(train_df)}, test rows: {len(test_df)}")
    print(f"temporal split boundary: {boundary.date()} (train < boundary <= test)")
    print(f"wrote {args.out_dir / 'disputes.csv'}")
    print(f"wrote {args.out_dir / 'debug.csv'}")


if __name__ == "__main__":
    main()
