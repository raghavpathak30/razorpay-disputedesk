"""Paired comparison of the two `customer_communication_log` -> `true_fraud`
feature-extraction arms: the LLM's typed fields vs. the TF-IDF baseline.

Makes **no network call** and needs **no API key**: the LLM arm is the
recorded 2026-09-01 run, committed at
`data/reference/llm_normalization_arm_n60_seed0.csv`. Both arms are scored on
the identical 60 items with the identical cross-validation folds, and the
difference is reported as a paired bootstrap over items.

Run as `python -m eval.run_extraction_comparison`.

Also reports the TF-IDF baseline alone at a larger n. That second figure is
*not* part of the paired comparison - it exists because n=60 is a small
sample, and a reader is entitled to know whether the baseline's own value at
n=60 is representative of the baseline or an artifact of the sample size. It
is cheap to compute (no API calls), so there is no reason not to.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from disputedesk.generator.config import GeneratorConfig
from eval.extraction_comparison import (
    auc_vs_chance,
    format_comparison,
    out_of_fold_probabilities,
    paired_auc_difference,
)
from eval.llm_normalization_quality import FEATURE_COLUMNS, auc_of_normalized_fields
from eval.tfidf_baseline import comms_and_true_fraud, tfidf_auc, tfidf_out_of_fold_probabilities

LLM_ARM_FIXTURE = Path("data/reference/llm_normalization_arm_n60_seed0.csv")
LLM_ARM_N_ROWS = 60
LLM_ARM_SEED = 0


def _load_llm_arm(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#")


def _assert_arms_are_the_same_items(llm_arm: pd.DataFrame, labels: np.ndarray) -> None:
    """Refuse to report a paired comparison whose pairing cannot be verified.

    The committed LLM arm carries its own `true_fraud` column; the generator
    is re-run here to produce the same label independently. If a generator
    change ever shifted the items under the recorded arm, row `i` of the two
    arms would no longer be the same dispute and every number below would be
    meaningless while still looking fine.
    """
    recorded = llm_arm["true_fraud"].to_numpy().astype(bool)
    if recorded.shape != labels.shape or not np.array_equal(recorded, labels):
        raise SystemExit(
            "the committed LLM arm no longer lines up with the generator's output: "
            f"{int((recorded != labels).sum()) if recorded.shape == labels.shape else 'length'} "
            "mismatch. The paired comparison cannot be reported until the LLM arm is "
            "re-measured against the current generator (see this module's docstring)."
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Paired LLM-vs-TF-IDF extraction comparison (no network, no API key)."
    )
    parser.add_argument("--fixture", type=Path, default=LLM_ARM_FIXTURE)
    parser.add_argument(
        "--baseline-only-n",
        type=int,
        default=3000,
        help="Rows for the standalone TF-IDF estimate reported alongside (not paired).",
    )
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args(argv)

    llm_arm = _load_llm_arm(args.fixture)
    logs, labels = comms_and_true_fraud(LLM_ARM_N_ROWS, LLM_ARM_SEED, GeneratorConfig())
    _assert_arms_are_the_same_items(llm_arm, labels)

    llm_features = llm_arm[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    llm_oof = out_of_fold_probabilities(llm_features, labels, random_state=args.random_state)
    tfidf_oof = tfidf_out_of_fold_probabilities(logs, labels, random_state=args.random_state)

    paired = paired_auc_difference(labels, tfidf_oof, llm_oof, random_state=args.random_state)

    print(
        f"items: n={len(labels)} (seed {LLM_ARM_SEED}), true_fraud prevalence {labels.mean():.4f}"
    )
    print(f"LLM arm: {args.fixture} (recorded run, no live call)")
    print()
    print("PAIRED - identical items, identical CV folds:")
    print(f"  {format_comparison(paired, 'TF-IDF', 'LLM typed fields')}")
    print()

    print("Each arm against chance (AUC 0.5), same items, same bootstrap:")
    for name, oof in (("TF-IDF", tfidf_oof), ("LLM typed fields", llm_oof)):
        chance = auc_vs_chance(labels, oof, random_state=args.random_state)
        verdict = "distinguishable" if chance.excludes_zero else "NOT distinguishable"
        print(
            f"  {name:<16}: AUC {chance.auc_a:.4f}, {chance.difference:+.4f} vs chance "
            f"(95% CI {chance.ci_low:+.4f} to {chance.ci_high:+.4f}) - {verdict}"
        )
    print()

    llm_folds = auc_of_normalized_fields(
        llm_arm[list(FEATURE_COLUMNS)].to_dict(orient="records"),
        labels,
        random_state=args.random_state,
    )
    tfidf_folds = tfidf_auc(logs, labels, random_state=args.random_state)
    print("Per-fold mean AUC (the shape both arms were originally reported in):")
    print(f"  TF-IDF          : {tfidf_folds['mean_auc']:.4f} (std {tfidf_folds['std_auc']:.4f})")
    print(f"  LLM typed fields: {llm_folds['mean_auc']:.4f} (std {llm_folds['std_auc']:.4f})")
    print()

    big_logs, big_labels = comms_and_true_fraud(args.baseline_only_n, LLM_ARM_SEED)
    big = tfidf_auc(big_logs, big_labels, random_state=args.random_state)
    print(
        f"TF-IDF baseline alone at n={big['n']} (NOT paired, not comparable to the "
        f"LLM arm - reported so the n=60 baseline value can be judged):"
    )
    print(f"  mean AUC {big['mean_auc']:.4f} (std {big['std_auc']:.4f})")


if __name__ == "__main__":
    main()
