"""The eval harness: one full generate -> split -> train -> score cycle per
seed, plus median/IQR aggregation across seeds. Runs in CI on a fixed seed set
(tests/test_eval_harness_regression.py) so a metric regression is caught at
commit time (PHASES.md Phase 2 gate), and via `eval/run_harness.py` for the
headline >=20-seed report.

Threshold note: Phase 2 has no policy engine yet (PHASES.md explicitly defers
it to Phase 3), so there is no expected-value threshold to report precision
and recall against. GENERATOR.md §5 itself rules out a fixed 0.5 threshold as
meaningless here (`p_max` is 0.75 and both population modes sit below 0.4, so
0.5 selects almost nothing). Until the policy engine exists, this harness uses
the *training-split label prevalence* as the operating threshold - a
documented placeholder, not a tuned or test-set-derived value - and flags it
as such in every report.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score

from disputedesk.features.matrix import build_feature_matrix
from disputedesk.generator.config import GeneratorConfig
from disputedesk.generator.pipeline import generate_dataset, temporal_split
from disputedesk.model.config import ModelConfig
from disputedesk.model.predict import predict_proba
from disputedesk.model.train import train
from eval.calibration import expected_calibration_error
from eval.oracle import oracle_pr_auc, prevalence_baseline

LABEL_COLUMN = "won_if_contested"


@dataclass(frozen=True)
class SeedResult:
    seed: int
    threshold: float
    # Named `_at_threshold`, not `precision`/`recall`, on purpose: these are
    # only meaningful together with `threshold` (submission checklist items 3
    # and 4 - SPEC.md §9 - will quote these numbers, and the placeholder
    # operating point they were measured at must travel with them everywhere
    # they're reported, not just in this row).
    precision_at_threshold: float
    recall_at_threshold: float
    model_pr_auc: float
    prevalence_baseline: float
    oracle_pr_auc: float
    calibration_error: float
    n_train: int
    n_test: int


def run_seed(
    seed: int,
    n_rows: int,
    generator_config: GeneratorConfig,
    model_config: ModelConfig,
) -> SeedResult:
    """One full cycle for a single seed. Every metric below is computed on the
    temporal test split only - nothing here ever touches `train_df` for
    scoring.
    """
    features_df, debug_df = generate_dataset(n_rows, seed, generator_config)
    train_df, test_df, _boundary = temporal_split(features_df, generator_config)

    X_train = build_feature_matrix(train_df)
    y_train = train_df[LABEL_COLUMN]
    X_test = build_feature_matrix(test_df)
    y_test = test_df[LABEL_COLUMN]

    model = train(X_train, y_train, model_config)
    predicted_p = predict_proba(model, X_test)

    threshold = float(y_train.mean())
    predicted_contest = predicted_p >= threshold

    test_debug_p = debug_df.loc[test_df.index, "p"].to_numpy()

    return SeedResult(
        seed=seed,
        threshold=threshold,
        precision_at_threshold=float(precision_score(y_test, predicted_contest, zero_division=0)),
        recall_at_threshold=float(recall_score(y_test, predicted_contest, zero_division=0)),
        model_pr_auc=float(average_precision_score(y_test, predicted_p)),
        prevalence_baseline=prevalence_baseline(y_test.to_numpy()),
        oracle_pr_auc=oracle_pr_auc(test_debug_p),
        calibration_error=expected_calibration_error(predicted_p, y_test.to_numpy()),
        n_train=len(train_df),
        n_test=len(test_df),
    )


def run_harness(
    seeds: list[int],
    n_rows: int,
    generator_config: GeneratorConfig | None = None,
    model_config: ModelConfig | None = None,
) -> pd.DataFrame:
    """Run `run_seed` across every seed and return one row per seed."""
    generator_config = generator_config or GeneratorConfig()
    model_config = model_config or ModelConfig()

    rows = [run_seed(seed, n_rows, generator_config, model_config) for seed in seeds]
    return pd.DataFrame([vars(r) for r in rows])


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Median and IQR (25th/75th percentile) of every numeric metric across
    seeds. Never report a single-seed number as a headline (CLAUDE.md
    invariant 3) - this is the function that turns per-seed rows into that
    headline.
    """
    numeric_cols = [c for c in results.columns if c != "seed"]
    summary = pd.DataFrame(
        {
            "median": results[numeric_cols].median(),
            "q25": results[numeric_cols].quantile(0.25),
            "q75": results[numeric_cols].quantile(0.75),
        }
    )
    summary.index.name = "metric"
    return summary


def fixed_seed_set(n_seeds: int, start: int = 0) -> list[int]:
    """The reproducible seed list used everywhere headline numbers are
    reported, so a `--seeds` typo can't quietly change what "20 seeds" means.
    """
    return list(np.arange(start, start + n_seeds))
