"""Requirement 7: the leakage guard is a first-class test, not a script. It
asserts no feature column is derivable from the label or from p. A deliberately
leaky control fixture (a column that copies p) proves the guard actually fires -
a guard that only ever passes is worthless.
"""

import numpy as np
import pandas as pd
import pytest

from disputedesk.generator.config import GeneratorConfig
from disputedesk.generator.pipeline import generate_dataset

MAX_ABS_CORR = 0.9
_NON_FEATURE_COLUMNS = {"id", "payment_id", "won_if_contested"}


def assert_no_leakage(
    feature_df: pd.DataFrame, target: pd.Series, max_abs_corr: float = MAX_ABS_CORR
) -> None:
    """Fails if any numeric/boolean column in feature_df is near-perfectly
    correlated with `target` - the shape a derivable ("won = f(columns)") column
    would take.
    """
    for col in feature_df.columns:
        if col in _NON_FEATURE_COLUMNS:
            continue
        series = feature_df[col]
        if not (pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)):
            continue
        corr = series.astype(float).corr(target.astype(float))
        if pd.isna(corr):
            continue
        assert abs(corr) < max_abs_corr, (
            f"column '{col}' correlates {corr:.3f} with the target - looks derivable"
        )


@pytest.fixture(scope="module")
def generated():
    config = GeneratorConfig()
    features_df, debug_df = generate_dataset(5000, seed=11, config=config)
    return features_df, debug_df


def test_real_features_do_not_leak_the_label(generated):
    features_df, _ = generated
    assert_no_leakage(features_df, target=features_df["won_if_contested"])


def test_real_features_do_not_leak_p(generated):
    features_df, debug_df = generated
    assert_no_leakage(features_df, target=debug_df["p"])


def test_guard_fires_on_a_column_that_copies_p(generated):
    features_df, debug_df = generated
    leaky_df = features_df.copy()
    leaky_df["leaky_copy_of_p"] = debug_df["p"].to_numpy()

    with pytest.raises(AssertionError, match="leaky_copy_of_p"):
        assert_no_leakage(leaky_df, target=debug_df["p"])


def test_guard_fires_on_a_column_that_copies_the_label():
    n = 200
    label = pd.Series(np.random.default_rng(0).random(n) < 0.3)
    df = pd.DataFrame({"harmless": np.arange(n), "leaky_copy_of_label": label.astype(float)})

    with pytest.raises(AssertionError, match="leaky_copy_of_label"):
        assert_no_leakage(df, target=label)
