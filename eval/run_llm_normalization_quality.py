"""Runs the LLM normalisation-quality measurement against the real Groq API
and reports it paired against the TF-IDF baseline, on the identical items.

Corrected 2026-09-03 (stale-number audit, remediation item A). This script
previously imported `TFIDF_BASELINE_AUC = 0.6371` from
`eval.llm_normalization_quality` and printed "beats baseline: YES/NO" against
it - the exact large-sample-baseline-vs-small-sample-LLM-run comparison
`DECISIONS.md`'s 2026-09-02 "TF-IDF baseline had no code" entry corrected.
The README's own "What would settle it" note told a future reader to run
this script to re-measure the LLM arm at a larger n; doing so would have
reproduced the defect the correction fixed. It no longer imports that
constant - see `test_no_stale_hardcoded_baseline_survives_in_the_run_script`
in `tests/test_eval_run_llm_normalization_quality_comparison.py`.

The TF-IDF arm is now computed here, at the same n and seed as the LLM
sample just collected (`eval.tfidf_baseline`, the module Phase 1 committed),
and the two are compared with `eval.extraction_comparison.paired_auc_difference`
- identical items, identical CV folds, a bootstrap interval, not a bare
point-estimate comparison.

Makes real network calls (the LLM arm only) and costs real (free-tier) API
usage - never run in CI, never imported by anything under `tests/` for that
reason. The pure comparison logic below (`paired_comparison_against_tfidf`)
takes an already-collected sample and makes no network call itself, so it is
unit-tested directly.

Run as `python -m eval.run_llm_normalization_quality`.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from disputedesk.evidence.llm import GroqHttpLLMClient
from disputedesk.generator.config import GeneratorConfig
from eval.extraction_comparison import (
    PairedAucDifference,
    auc_vs_chance,
    format_comparison,
    out_of_fold_probabilities,
    paired_auc_difference,
)
from eval.llm_normalization_quality import (
    FEATURE_COLUMNS,
    auc_of_normalized_fields,
    run_llm_normalization_sample,
)
from eval.tfidf_baseline import comms_and_true_fraud, tfidf_auc, tfidf_out_of_fold_probabilities


def paired_comparison_against_tfidf(
    sample: pd.DataFrame, n_rows: int, seed: int, random_state: int = 0
) -> PairedAucDifference:
    """Pair `sample` (an LLM-arm frame from `run_llm_normalization_sample`)
    against a freshly-computed TF-IDF baseline on the same `n_rows`/`seed`
    generator items.

    Regenerates the dataset independently and checks `sample["true_fraud"]`
    matches it row for row before comparing anything - the same safeguard
    `eval/run_extraction_comparison.py` uses for the committed n=60 fixture,
    here applied to a fresh sample instead of a committed one. Without it, a
    caller passing a mismatched `n_rows`/`seed` would get a comparison whose
    two arms are not actually the same items and would have no way to tell.
    """
    logs, labels = comms_and_true_fraud(n_rows, seed, GeneratorConfig())
    recorded = sample["true_fraud"].to_numpy().astype(bool)
    if recorded.shape != labels.shape or not np.array_equal(recorded, labels):
        raise ValueError(
            "sample['true_fraud'] does not match the generator's output for "
            f"n_rows={n_rows}, seed={seed} - the two arms are not paired"
        )

    llm_features = sample[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    llm_oof = out_of_fold_probabilities(llm_features, labels, random_state=random_state)
    tfidf_oof = tfidf_out_of_fold_probabilities(logs, labels, random_state=random_state)

    return paired_auc_difference(labels, tfidf_oof, llm_oof, random_state=random_state)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Measure LLM communication-log normalisation quality, paired against TF-IDF."
    )
    parser.add_argument("--n-rows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=5.0,
        help="Delay between LLM calls, to stay under Groq free-tier rate limits.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/eval"))
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args(argv)

    llm_client = GroqHttpLLMClient()
    sample = run_llm_normalization_sample(
        args.n_rows,
        args.seed,
        GeneratorConfig(),
        llm_client,
        sleep_seconds=args.sleep_seconds,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.out_dir / "llm_normalization_quality_sample.csv", index=False)

    feature_rows = sample[list(FEATURE_COLUMNS)].to_dict(orient="records")
    llm_result = auc_of_normalized_fields(feature_rows, sample["true_fraud"], random_state=0)

    print(f"n={llm_result['n']}, true_fraud prevalence={llm_result['prevalence']:.3f}")
    print(f"human_review_required rate: {sample['human_review_required'].mean():.3f}")
    print(f"LLM fold AUCs: {[round(a, 4) for a in llm_result['fold_aucs']]}")
    print(
        f"LLM-normalized-fields AUC (mean of 5-fold CV): {llm_result['mean_auc']:.4f} "
        f"(std {llm_result['std_auc']:.4f})"
    )
    print()

    paired = paired_comparison_against_tfidf(
        sample, args.n_rows, args.seed, random_state=args.random_state
    )
    print("PAIRED against TF-IDF - identical items, identical CV folds:")
    print(f"  {format_comparison(paired, 'TF-IDF', 'LLM typed fields')}")
    print()
    print("Each arm against chance (AUC 0.5), same items, same bootstrap:")
    logs, labels = comms_and_true_fraud(args.n_rows, args.seed, GeneratorConfig())
    llm_oof = out_of_fold_probabilities(
        sample[list(FEATURE_COLUMNS)].to_numpy(dtype=float), labels, random_state=args.random_state
    )
    tfidf_oof = tfidf_out_of_fold_probabilities(logs, labels, random_state=args.random_state)
    for name, oof in (("TF-IDF", tfidf_oof), ("LLM typed fields", llm_oof)):
        chance = auc_vs_chance(labels, oof, random_state=args.random_state)
        verdict = "distinguishable" if chance.excludes_zero else "NOT distinguishable"
        print(
            f"  {name:<16}: AUC {chance.auc_a:.4f}, {chance.difference:+.4f} vs chance "
            f"(95% CI {chance.ci_low:+.4f} to {chance.ci_high:+.4f}) - {verdict}"
        )
    print()
    tfidf_result = tfidf_auc(logs, labels, random_state=args.random_state)
    print(
        f"TF-IDF per-fold mean AUC (same items): {tfidf_result['mean_auc']:.4f} "
        f"(std {tfidf_result['std_auc']:.4f})"
    )
    print()
    print(f"wrote {args.out_dir / 'llm_normalization_quality_sample.csv'}")
    print(
        "to make this a durable, reproducible-without-a-key comparison: commit the sample "
        "CSV above to data/reference/ (see llm_normalization_arm_n60_seed0.csv for the "
        "n=60 precedent) and add it to eval/run_extraction_comparison.py."
    )


if __name__ == "__main__":
    main()
