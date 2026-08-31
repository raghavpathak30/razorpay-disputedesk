"""Requirement 6 & 8: temporal split by timestamp, not by row order or random
assignment, and no overlap between train and test.
"""

from disputedesk.generator.config import GeneratorConfig
from disputedesk.generator.pipeline import generate_dataset, temporal_split


def test_split_has_no_timestamp_overlap():
    config = GeneratorConfig()
    features_df, _ = generate_dataset(3000, seed=9, config=config)

    train_df, test_df, boundary = temporal_split(features_df, config)

    assert len(train_df) + len(test_df) == len(features_df)
    assert (train_df["purchase_ts"] < boundary).all()
    assert (test_df["purchase_ts"] >= boundary).all()
    assert train_df["purchase_ts"].max() < test_df["purchase_ts"].min()


def test_split_is_not_a_row_count_split():
    """A time-based split need not be (and here should not be) an even row-count
    split, since disputes are drawn uniformly over the window but the split
    boundary is a fixed calendar point, not a fixed row index."""
    config = GeneratorConfig()
    features_df, _ = generate_dataset(3000, seed=9, config=config)

    train_df, test_df, _ = temporal_split(features_df, config)
    expected_train_row_share = config.train_window_months / config.simulation_window_months

    actual_train_row_share = len(train_df) / len(features_df)
    assert abs(actual_train_row_share - expected_train_row_share) < 0.05
