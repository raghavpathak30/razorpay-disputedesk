"""Batch form of `build_features`, for model training/inference. Still pure: a
DataFrame in, a DataFrame out, no file access and no network - `build_features`
applied row-wise, nothing more.
"""

import pandas as pd

from disputedesk.features.build import FEATURE_COLUMNS, build_features


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Apply `build_features` to every row of `df` and return the model-facing
    feature matrix, columns in `FEATURE_COLUMNS` order.
    """
    rows = [build_features(row) for row in df.to_dict(orient="records")]
    return pd.DataFrame(rows, columns=list(FEATURE_COLUMNS), index=df.index)
