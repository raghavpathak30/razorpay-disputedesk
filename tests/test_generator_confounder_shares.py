"""Requirement 8: confounder population shares match config within tolerance."""

from disputedesk.generator.config import GeneratorConfig
from disputedesk.generator.latents import (
    ACCOUNT_TAKEOVER_CONFOUNDER,
    FRAUD_NONCONFOUNDER,
    GENUINE_NONCONFOUNDER,
    TRAVELER_CONFOUNDER,
)
from disputedesk.generator.pipeline import generate_dataset

TOLERANCE = 0.03  # absolute, generous enough for n=20000 sampling noise


def test_confounder_shares_within_tolerance_of_config():
    config = GeneratorConfig()
    _, debug_df = generate_dataset(20000, seed=5, config=config)

    fraud_mask = debug_df["true_fraud"]
    genuine_mask = ~fraud_mask

    ato_share_of_fraud = (
        debug_df.loc[fraud_mask, "component"] == ACCOUNT_TAKEOVER_CONFOUNDER
    ).mean()
    traveler_share_of_genuine = (
        debug_df.loc[genuine_mask, "component"] == TRAVELER_CONFOUNDER
    ).mean()

    assert abs(ato_share_of_fraud - config.account_takeover_share_of_fraud) < TOLERANCE
    assert abs(traveler_share_of_genuine - config.traveler_share_of_genuine) < TOLERANCE

    # every row belongs to exactly one of the four named components
    known = {
        ACCOUNT_TAKEOVER_CONFOUNDER,
        FRAUD_NONCONFOUNDER,
        GENUINE_NONCONFOUNDER,
        TRAVELER_CONFOUNDER,
    }
    assert set(debug_df["component"].unique()) <= known


def test_true_fraud_rate_within_configured_drift_range():
    config = GeneratorConfig()
    _, debug_df = generate_dataset(20000, seed=5, config=config)

    fraud_rate = debug_df["true_fraud"].mean()
    lo = min(config.true_fraud_rate_month0, config.true_fraud_rate_month_last) - TOLERANCE
    hi = max(config.true_fraud_rate_month0, config.true_fraud_rate_month_last) + TOLERANCE
    assert lo <= fraud_rate <= hi
