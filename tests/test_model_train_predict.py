"""Model tests: trains only on the training split, predicts P(win) (never a
decision), and never sees `p`, any latent, or the label at inference.
"""

import numpy as np

from disputedesk.features.matrix import build_feature_matrix
from disputedesk.generator.config import GeneratorConfig
from disputedesk.generator.pipeline import generate_dataset, temporal_split
from disputedesk.model.config import ModelConfig
from disputedesk.model.predict import predict_proba
from disputedesk.model.train import train


def _split_features(n_rows=4000, seed=7):
    config = GeneratorConfig()
    features_df, debug_df = generate_dataset(n_rows, seed, config)
    train_df, test_df, _boundary = temporal_split(features_df, config)
    return train_df, test_df, debug_df


def test_predict_proba_is_a_probability_never_a_decision():
    train_df, test_df, _debug_df = _split_features()
    X_train = build_feature_matrix(train_df)
    y_train = train_df["won_if_contested"]
    X_test = build_feature_matrix(test_df)

    model = train(X_train, y_train, ModelConfig(n_estimators=50))
    predicted_p = predict_proba(model, X_test)

    assert predicted_p.dtype.kind == "f"
    assert (predicted_p >= 0).all()
    assert (predicted_p <= 1).all()
    assert predicted_p.shape[0] == len(X_test)


def test_model_input_never_contains_debug_or_label_columns():
    train_df, _test_df, _debug_df = _split_features()
    X_train = build_feature_matrix(train_df)

    forbidden = {"p", "true_fraud", "component", "epsilon", "won_if_contested", "id"}
    assert forbidden.isdisjoint(X_train.columns)


def test_model_beats_random_on_holdout_pr_auc():
    """Sanity floor, not a headline number: a trained model should clear the
    prevalence baseline on the temporal holdout by some margin. The real,
    multi-seed comparison lives in eval/ (PHASES.md Phase 2 gate).
    """
    from sklearn.metrics import average_precision_score

    train_df, test_df, _debug_df = _split_features(n_rows=8000, seed=3)
    X_train = build_feature_matrix(train_df)
    y_train = train_df["won_if_contested"]
    X_test = build_feature_matrix(test_df)
    y_test = test_df["won_if_contested"]

    model = train(X_train, y_train, ModelConfig())
    predicted_p = predict_proba(model, X_test)

    pr_auc = average_precision_score(y_test, predicted_p)
    prevalence = y_test.mean()
    assert pr_auc > prevalence


def test_train_is_deterministic_given_the_same_seed_and_config():
    train_df, test_df, _debug_df = _split_features()
    X_train = build_feature_matrix(train_df)
    y_train = train_df["won_if_contested"]
    X_test = build_feature_matrix(test_df)

    model_a = train(X_train, y_train, ModelConfig(random_state=5))
    model_b = train(X_train, y_train, ModelConfig(random_state=5))

    np.testing.assert_allclose(predict_proba(model_a, X_test), predict_proba(model_b, X_test))
