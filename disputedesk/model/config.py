"""LightGBM hyperparameters. One place, so a headline number can always be traced
to the parameters that produced it, per CLAUDE.md's config-module convention.
"""

from pydantic import BaseModel, ConfigDict


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_estimators: int = 200
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    random_state: int = 0
    verbosity: int = -1
