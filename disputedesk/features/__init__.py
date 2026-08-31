"""Feature builder: pure functions turning dispute/order context into model features, no I/O."""

from disputedesk.features.build import (
    CARD_NETWORKS,
    CATEGORICAL_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    REASON_CODES,
    build_features,
)
from disputedesk.features.matrix import build_feature_matrix

__all__ = [
    "CARD_NETWORKS",
    "CATEGORICAL_FEATURE_COLUMNS",
    "FEATURE_COLUMNS",
    "REASON_CODES",
    "build_feature_matrix",
    "build_features",
]
