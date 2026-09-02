"""Single-feature ablation of the win-probability model (Phase 2 addendum,
item B, 2026-09-03).

Why this module exists. The rebuilt leakage guard's discrimination-ceiling
check (`eval/leakage.py`) measures each feature's univariate AUC as a
byproduct of checking it is not a leak. That measurement showed
`ip_geo_billing_distance_km` alone reaches 69.9% of the Bayes ceiling - not a
leak (98% is the flag threshold, and the guard's own shuffled-label control
proves it isn't one), but strong enough that a fair question follows: against
a full-feature policy advantage of only ≈0.66% over "contest everything" at
the configured cost, how much of that advantage is one geo feature carrying,
and how much is the other eleven adding on top of it?

This module answers it by re-running the exact same business harness -
same seeds, same paired estimator, same cost sweep
(`eval.cost_sensitivity.summarize_sweep`) - three times, restricting the
model to progressively more of the ranked feature set: the top feature alone,
the top three, and the full twelve. `predictions_for_feature_subset` fits its
own `lgb.LGBMClassifier` directly rather than calling
`disputedesk.model.train.train`, because that function hardcodes
`categorical_feature` against the *full* declared feature set
(`disputedesk.features.build.CATEGORICAL_FEATURE_COLUMNS`) and would be asked
for a categorical column not present in a restricted `X_train` - this module
filters that list to the columns actually being used instead. Production
training code is untouched.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from disputedesk.features.build import CATEGORICAL_FEATURE_COLUMNS, FEATURE_COLUMNS
from disputedesk.features.matrix import build_feature_matrix
from disputedesk.generator.config import GeneratorConfig
from disputedesk.generator.pipeline import generate_dataset, temporal_split
from disputedesk.model.config import ModelConfig
from disputedesk.model.predict import predict_proba
from eval.cost_sensitivity import score_predictions_across_costs
from eval.harness import LABEL_COLUMN

# The guard's own measured ranking on its fixed fixture (n=5,000, seed=11,
# `tests/test_generator_leakage_guard.py`'s `generated` fixture) - not a
# hand-picked guess. Pinned by
# `tests/test_eval_ablation.py::test_top_feature_constants_match_the_leakage_reports_ranking`
# against a live `eval.leakage.leakage_report` call on that same fixture, so
# the two cannot silently diverge.
TOP_1_FEATURE = "ip_geo_billing_distance_km"
TOP_3_FEATURES: tuple[str, ...] = (
    "ip_geo_billing_distance_km",
    "prior_order_count",
    "avs_match",
)


def predictions_for_feature_subset(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: tuple[str, ...],
    model_config: ModelConfig,
) -> np.ndarray:
    """Train on exactly `feature_columns` (a subset of
    `disputedesk.features.build.FEATURE_COLUMNS`) and return `P(win)` on
    `test_df`.

    Raises on an undeclared column name rather than silently training on
    whatever happened to be passed - the whole point of an ablation is that
    the feature set is exactly what it claims to be.
    """
    unknown = set(feature_columns) - set(FEATURE_COLUMNS)
    if unknown:
        raise ValueError(f"not a declared feature: {sorted(unknown)}")

    X_train = build_feature_matrix(train_df)[list(feature_columns)]
    y_train = train_df[LABEL_COLUMN]
    X_test = build_feature_matrix(test_df)[list(feature_columns)]

    categorical = [c for c in CATEGORICAL_FEATURE_COLUMNS if c in feature_columns]
    model = lgb.LGBMClassifier(
        n_estimators=model_config.n_estimators,
        learning_rate=model_config.learning_rate,
        num_leaves=model_config.num_leaves,
        max_depth=model_config.max_depth,
        min_child_samples=model_config.min_child_samples,
        subsample=model_config.subsample,
        colsample_bytree=model_config.colsample_bytree,
        reg_lambda=model_config.reg_lambda,
        random_state=model_config.random_state,
        verbosity=model_config.verbosity,
    )
    model.fit(X_train, y_train, categorical_feature=categorical)
    return predict_proba(model, X_test)


def sweep_feature_subset_seed(
    seed: int,
    n_rows: int,
    costs: list[float],
    feature_columns: tuple[str, ...],
    generator_config: GeneratorConfig,
    model_config: ModelConfig,
    low_confidence_band: tuple[float, float],
) -> list[dict]:
    """One seed's generate/split/train(restricted)/predict cycle, scored
    across every swept cost via the same
    `eval.cost_sensitivity.score_predictions_across_costs` the full-feature
    sweep uses - identical scoring, the only difference is which columns the
    model saw during training.
    """
    features_df, _debug_df = generate_dataset(n_rows, seed, generator_config)
    train_df, test_df, _boundary = temporal_split(features_df, generator_config)

    predicted_p = predictions_for_feature_subset(train_df, test_df, feature_columns, model_config)
    amount = test_df["amount"].to_numpy()
    won_if_contested = test_df[LABEL_COLUMN].to_numpy()

    return score_predictions_across_costs(
        seed, predicted_p, amount, won_if_contested, costs, low_confidence_band
    )


def sweep_feature_subset(
    seeds: list[int],
    n_rows: int,
    costs: list[float],
    feature_columns: tuple[str, ...],
    generator_config: GeneratorConfig | None = None,
    model_config: ModelConfig | None = None,
    low_confidence_band: tuple[float, float] = (0.45, 0.55),
) -> pd.DataFrame:
    """One row per (seed, cost), same shape as
    `eval.cost_sensitivity.sweep_representment_cost` - pass straight to
    `eval.cost_sensitivity.summarize_sweep` for the paired estimator.
    """
    generator_config = generator_config or GeneratorConfig()
    model_config = model_config or ModelConfig()

    rows: list[dict] = []
    for seed in seeds:
        rows.extend(
            sweep_feature_subset_seed(
                seed,
                n_rows,
                costs,
                feature_columns,
                generator_config,
                model_config,
                low_confidence_band,
            )
        )
    return pd.DataFrame(rows)
