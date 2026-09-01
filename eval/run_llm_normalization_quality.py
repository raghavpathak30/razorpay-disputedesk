"""Runs the LLM normalisation-quality measurement against the real Groq API
and reports AUC(normalized fields, true_fraud) next to the recorded 0.6371
TF-IDF baseline (DECISIONS.md, 2026-08-31 "Generator calibration").

Makes real network calls and costs real (free-tier) API usage - never run
in CI, never imported by anything under `tests/`.

Run as `python -m eval.run_llm_normalization_quality`.
"""

import argparse
from pathlib import Path

from disputedesk.evidence.llm import GroqHttpLLMClient
from disputedesk.generator.config import GeneratorConfig
from eval.llm_normalization_quality import (
    FEATURE_COLUMNS,
    TFIDF_BASELINE_AUC,
    auc_of_normalized_fields,
    run_llm_normalization_sample,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Measure LLM communication-log normalisation quality vs. the TF-IDF baseline."
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
    args = parser.parse_args(argv)

    llm_client = GroqHttpLLMClient()
    sample = run_llm_normalization_sample(
        args.n_rows,
        args.seed,
        GeneratorConfig(),
        llm_client,
        sleep_seconds=args.sleep_seconds,
    )

    feature_rows = sample[list(FEATURE_COLUMNS)].to_dict(orient="records")
    result = auc_of_normalized_fields(feature_rows, sample["true_fraud"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.out_dir / "llm_normalization_quality_sample.csv", index=False)

    print(f"n={result['n']}, true_fraud prevalence={result['prevalence']:.3f}")
    print(f"human_review_required rate: {sample['human_review_required'].mean():.3f}")
    print(f"fold AUCs: {[round(a, 4) for a in result['fold_aucs']]}")
    print(
        f"LLM-normalized-fields AUC (mean of 5-fold CV): {result['mean_auc']:.4f} "
        f"(std {result['std_auc']:.4f})"
    )
    print(f"TF-IDF + logistic regression baseline (recorded, 5-fold CV): {TFIDF_BASELINE_AUC}")
    beats = result["mean_auc"] > TFIDF_BASELINE_AUC
    print(f"beats baseline: {'YES' if beats else 'NO'}")
    print()
    print(f"wrote {args.out_dir / 'llm_normalization_quality_sample.csv'}")


if __name__ == "__main__":
    main()
