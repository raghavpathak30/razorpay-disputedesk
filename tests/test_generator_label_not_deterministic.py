"""Requirement 8: the label is never a deterministic function of any feature
subset. CLAUDE.md invariant 1: if `won = f(features)` held, grouping by any
feature combination would make every group pure (one label value only). With p
bounded away from {0, 1} (GENERATOR.md §4) this must not happen at any realistic
sample size.
"""

from disputedesk.generator.config import GeneratorConfig
from disputedesk.generator.pipeline import generate_dataset

_BOOLEAN_FEATURES = [
    "avs_match",
    "cvv_match",
    "device_fingerprint_known",
    "delivery_confirmed",
]


def test_label_not_deterministic_given_single_boolean_feature():
    config = GeneratorConfig()
    features_df, _ = generate_dataset(5000, seed=3, config=config)

    for col in _BOOLEAN_FEATURES:
        labels_by_value = features_df.groupby(col)["won_if_contested"].nunique()
        assert (labels_by_value > 1).all(), f"{col} alone determines the label"


def test_label_not_deterministic_given_full_boolean_combination():
    config = GeneratorConfig()
    features_df, _ = generate_dataset(5000, seed=3, config=config)

    grouped = features_df.groupby(_BOOLEAN_FEATURES)["won_if_contested"].agg(["nunique", "size"])
    well_populated = grouped[grouped["size"] >= 20]
    assert not well_populated.empty
    assert (well_populated["nunique"] > 1).all(), (
        "some well-populated combination of boolean features perfectly determines the label"
    )
