"""Permutation importance only - never LightGBM's gain importance (see
eval/importance.py's docstring for why: gain ranks the pure-noise
`checkout_hour_of_day` control fourth on this dataset).
"""

from disputedesk.features.build import FEATURE_COLUMNS
from disputedesk.features.matrix import build_feature_matrix
from disputedesk.generator.config import GeneratorConfig
from disputedesk.generator.pipeline import generate_dataset, temporal_split
from disputedesk.model.config import ModelConfig
from disputedesk.model.train import train
from eval.importance import permutation_importance_report


def test_permutation_importance_report_covers_every_feature():
    config = GeneratorConfig()
    features_df, _debug_df = generate_dataset(4000, seed=1, config=config)
    train_df, test_df, _boundary = temporal_split(features_df, config)

    X_train = build_feature_matrix(train_df)
    y_train = train_df["won_if_contested"]
    X_test = build_feature_matrix(test_df)
    y_test = test_df["won_if_contested"]

    model = train(X_train, y_train, ModelConfig(n_estimators=50))
    report = permutation_importance_report(model, X_test, y_test, n_repeats=3)

    assert set(report["feature"]) == set(FEATURE_COLUMNS)
    assert len(report) == len(FEATURE_COLUMNS)
    # Sorted most-important first.
    assert list(report["importance_mean"]) == sorted(report["importance_mean"], reverse=True)
