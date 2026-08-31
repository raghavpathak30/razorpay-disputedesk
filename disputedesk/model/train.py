"""Trains the win-probability model on the temporal training split only. The
model outputs `P(win)`; it makes no contest/accept/escalate decision (SPEC.md
§2, §4 - that is `policy/`'s job, not built yet in Phase 2).
"""

import lightgbm as lgb
import pandas as pd

from disputedesk.features.build import CATEGORICAL_FEATURE_COLUMNS
from disputedesk.model.config import ModelConfig


def train(X_train: pd.DataFrame, y_train: pd.Series, config: ModelConfig) -> lgb.LGBMClassifier:
    """Fit a LightGBM classifier on the training split. `X_train` must be the
    output of `build_feature_matrix` - never raw generator/debug columns, and
    never anything from the test split.
    """
    model = lgb.LGBMClassifier(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        max_depth=config.max_depth,
        min_child_samples=config.min_child_samples,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_lambda=config.reg_lambda,
        random_state=config.random_state,
        verbosity=config.verbosity,
    )
    model.fit(
        X_train,
        y_train,
        categorical_feature=list(CATEGORICAL_FEATURE_COLUMNS),
    )
    return model
