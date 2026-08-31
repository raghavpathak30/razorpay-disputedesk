"""Requirement 4: same seed, byte-identical output."""

import pandas as pd

from disputedesk.generator.config import GeneratorConfig
from disputedesk.generator.pipeline import generate_dataset


def test_same_seed_produces_identical_frames():
    config = GeneratorConfig()
    features_a, debug_a = generate_dataset(500, seed=7, config=config)
    features_b, debug_b = generate_dataset(500, seed=7, config=config)

    pd.testing.assert_frame_equal(features_a, features_b)
    pd.testing.assert_frame_equal(debug_a, debug_b)


def test_different_seeds_diverge():
    config = GeneratorConfig()
    features_a, _ = generate_dataset(500, seed=1, config=config)
    features_b, _ = generate_dataset(500, seed=2, config=config)

    assert not features_a["amount"].equals(features_b["amount"])
