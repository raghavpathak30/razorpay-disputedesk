"""p stays inside the configured band, and generated rows validate against the
schema in disputedesk/generator/schema.py (generate_dataset already validates
every row internally; this test checks the boundary values directly).
"""

from disputedesk.generator.config import GeneratorConfig
from disputedesk.generator.pipeline import generate_dataset


def test_p_is_always_within_the_configured_band():
    config = GeneratorConfig()
    _, debug_df = generate_dataset(5000, seed=13, config=config)

    assert debug_df["p"].min() >= config.p_min
    assert debug_df["p"].max() <= config.p_max


def test_e_p_is_close_to_the_documented_target():
    config = GeneratorConfig()
    _, debug_df = generate_dataset(20000, seed=13, config=config)

    e_p = debug_df["p"].mean()
    assert 0.20 <= e_p <= 0.30, (
        f"E[p]={e_p:.4f} drifted well outside the GENERATOR.md §4 target range"
    )


def test_reason_code_is_one_of_the_confirmed_codes():
    config = GeneratorConfig()
    features_df, _ = generate_dataset(2000, seed=13, config=config)

    assert set(features_df["reason_code"].unique()) <= set(config.reason_codes)
