"""GENERATOR.md §2 steps 1-4: timestamps, the mixture component, the continuous
latent causes, and the irreducible residual. Every latent here is drawn from the
seeded `rng` passed in explicitly — nothing here touches global random state, and
nothing here reads `p` or the label (both come later, in probability.py / pipeline.py).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from disputedesk.generator.config import GeneratorConfig

FRAUD_NONCONFOUNDER = "fraud_nonconfounder"
ACCOUNT_TAKEOVER_CONFOUNDER = "account_takeover_confounder"  # §6B
GENUINE_NONCONFOUNDER = "genuine_nonconfounder"
TRAVELER_CONFOUNDER = "traveler_confounder"  # §6A

SIMULATION_START = pd.Timestamp("2024-01-01")  # implementation detail per §7, not a causal choice


@dataclass(frozen=True)
class LatentBatch:
    purchase_ts: np.ndarray  # datetime64[ns]
    period_index: np.ndarray  # int, month index 0..window-1
    true_fraud: np.ndarray  # bool, L1
    component: np.ndarray  # str, mixture component from §6
    authentication_strength: np.ndarray  # float [0,1], L2
    relationship_genuineness: np.ndarray  # float [0,1], L3
    delivery_provability: np.ndarray  # float [0,1], L4
    filing_delay_days: np.ndarray  # float >=0, L5
    dispute_propensity: np.ndarray  # float [0,1], L7
    reason_subtype: np.ndarray  # str, L6
    epsilon: np.ndarray  # float, irreducible residual


def draw_timestamps(
    rng: np.random.Generator, n: int, config: GeneratorConfig
) -> tuple[np.ndarray, np.ndarray]:
    """§2 step 1. Uniform draw across the simulation window; no seasonality term
    is modeled since GENERATOR.md §7 flags it as non-causal, realism-only."""
    window_days = config.simulation_window_months * 30
    offsets_days = rng.uniform(0.0, window_days, size=n)
    purchase_ts = (SIMULATION_START + pd.to_timedelta(offsets_days, unit="D")).values
    period_index = np.minimum((offsets_days / 30).astype(int), config.simulation_window_months - 1)
    return purchase_ts, period_index


def draw_mixture_component(
    rng: np.random.Generator, n: int, period_index: np.ndarray, config: GeneratorConfig
) -> tuple[np.ndarray, np.ndarray]:
    """§2 step 2. true_fraud rate drifts by period_index (§7); the confounder split
    within each true_fraud class is drawn independently per row."""
    window = config.simulation_window_months - 1
    fraud_rate = config.true_fraud_rate_month0 + (period_index / window) * (
        config.true_fraud_rate_month_last - config.true_fraud_rate_month0
    )
    true_fraud = rng.random(n) < fraud_rate

    confounder_share = np.where(
        true_fraud, config.account_takeover_share_of_fraud, config.traveler_share_of_genuine
    )
    is_confounder = rng.random(n) < confounder_share

    component = np.select(
        [
            true_fraud & is_confounder,
            true_fraud & ~is_confounder,
            ~true_fraud & is_confounder,
            ~true_fraud & ~is_confounder,
        ],
        [
            ACCOUNT_TAKEOVER_CONFOUNDER,
            FRAUD_NONCONFOUNDER,
            TRAVELER_CONFOUNDER,
            GENUINE_NONCONFOUNDER,
        ],
        default="unreachable",
    )
    return true_fraud, component


def _component_beta_draw(
    rng: np.random.Generator,
    n: int,
    component: np.ndarray,
    params_by_component: dict[str, tuple[float, float]],
) -> np.ndarray:
    a = np.select(
        [component == key for key in params_by_component],
        [val[0] for val in params_by_component.values()],
    )
    b = np.select(
        [component == key for key in params_by_component],
        [val[1] for val in params_by_component.values()],
    )
    return rng.beta(a, b, size=n)


def _auth_strength_params(config: GeneratorConfig) -> dict[str, tuple[float, float]]:
    return {
        FRAUD_NONCONFOUNDER: config.auth_strength_beta_fraud_nonconfounder,
        ACCOUNT_TAKEOVER_CONFOUNDER: config.auth_strength_beta_account_takeover,
        GENUINE_NONCONFOUNDER: config.auth_strength_beta_genuine_nonconfounder,
        TRAVELER_CONFOUNDER: config.auth_strength_beta_traveler,
    }


def _relationship_params(config: GeneratorConfig) -> dict[str, tuple[float, float]]:
    return {
        FRAUD_NONCONFOUNDER: config.relationship_beta_fraud_nonconfounder,
        ACCOUNT_TAKEOVER_CONFOUNDER: config.relationship_beta_account_takeover,
        GENUINE_NONCONFOUNDER: config.relationship_beta_genuine_nonconfounder,
        TRAVELER_CONFOUNDER: config.relationship_beta_traveler,
    }


def _draw_filing_delay_days(
    rng: np.random.Generator, true_fraud: np.ndarray, config: GeneratorConfig
) -> np.ndarray:
    mean_days = np.where(
        true_fraud, config.filing_delay_mean_true_fraud_days, config.filing_delay_mean_genuine_days
    )
    shape = config.filing_delay_gamma_shape
    return rng.gamma(shape, scale=mean_days / shape)


def draw_continuous_latents(
    rng: np.random.Generator,
    n: int,
    true_fraud: np.ndarray,
    component: np.ndarray,
    config: GeneratorConfig,
) -> dict[str, np.ndarray]:
    """§2 step 3. Each continuous latent's distribution is conditioned on the
    component drawn in step 2 — confounding is structural, per §6's closing note."""
    authentication_strength = _component_beta_draw(rng, n, component, _auth_strength_params(config))
    relationship_genuineness = _component_beta_draw(rng, n, component, _relationship_params(config))
    delivery_provability = rng.beta(*config.delivery_provability_beta, size=n)
    filing_delay_days = _draw_filing_delay_days(rng, true_fraud, config)
    dispute_propensity = rng.beta(
        *config.dispute_propensity_beta, size=n
    )  # §1 L7: independent of component
    reason_subtype = rng.choice(np.array(config.reason_codes), size=n)

    return {
        "authentication_strength": authentication_strength,
        "relationship_genuineness": relationship_genuineness,
        "delivery_provability": delivery_provability,
        "filing_delay_days": filing_delay_days,
        "dispute_propensity": dispute_propensity,
        "reason_subtype": reason_subtype,
    }


def draw_epsilon(rng: np.random.Generator, n: int, config: GeneratorConfig) -> np.ndarray:
    """§2 step 4. Irreducible residual, independent of every named latent."""
    return rng.normal(0.0, config.epsilon_sigma, size=n)
