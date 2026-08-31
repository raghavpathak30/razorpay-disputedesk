"""Scores rows with a trained model. Returns `P(win)` only - never a decision."""

import lightgbm as lgb
import numpy as np
import pandas as pd


def predict_proba(model: lgb.LGBMClassifier, X: pd.DataFrame) -> np.ndarray:
    """`P(win_if_contested)` for each row of `X`, i.e. the probability of the
    `True` class.
    """
    return model.predict_proba(X)[:, 1]
