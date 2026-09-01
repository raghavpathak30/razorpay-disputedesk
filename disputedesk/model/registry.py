"""Where a trained model comes from for live serving (the webhook, the demo
script). No persisted model artifact/registry exists in this project (not
listed in SPEC.md or PHASES.md - out of scope for this phase); building one
would be new scope this session was not asked for.

`get_default_model_bundle` trains one model, once per process
(`lru_cache`), from the exact same generate -> temporal-split -> train
pipeline `eval/harness.py`'s `run_seed_pipeline` already uses for every
headline number in this project - reused for serving, not a second training
path. `MODEL_VERSION` names the fixed seed and model config a decision was
scored with, so every audit row is traceable to what produced its `P(win)`,
per PHASES.md Phase 4's audit-row requirement; it should change if
`ModelConfig`, `GeneratorConfig`, or `DEFAULT_TRAINING_SEED` change.
"""

from dataclasses import dataclass
from functools import lru_cache

import lightgbm as lgb

from disputedesk.features.matrix import build_feature_matrix
from disputedesk.generator.config import GeneratorConfig
from disputedesk.generator.pipeline import generate_dataset, temporal_split
from disputedesk.model.config import ModelConfig
from disputedesk.model.train import train

DEFAULT_TRAINING_SEED = 42
DEFAULT_TRAINING_N_ROWS = 15000
MODEL_VERSION = f"lgbm-config-v1-seed{DEFAULT_TRAINING_SEED}"

LABEL_COLUMN = "won_if_contested"


@dataclass(frozen=True)
class ModelBundle:
    model: lgb.LGBMClassifier
    version: str


@lru_cache
def get_default_model_bundle(n_rows: int = DEFAULT_TRAINING_N_ROWS) -> ModelBundle:
    generator_config = GeneratorConfig()
    model_config = ModelConfig()

    features_df, _debug_df = generate_dataset(n_rows, DEFAULT_TRAINING_SEED, generator_config)
    train_df, _test_df, _boundary = temporal_split(features_df, generator_config)

    X_train = build_feature_matrix(train_df)
    y_train = train_df[LABEL_COLUMN]
    model = train(X_train, y_train, model_config)

    return ModelBundle(model=model, version=MODEL_VERSION)
