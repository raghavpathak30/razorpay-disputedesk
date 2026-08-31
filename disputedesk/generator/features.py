"""GENERATOR.md §2 step 6: every observable feature, derived only from the latents
in latents.py (never from p, never from the label), each with its own independent
sensor noise. Also the pure-noise negative-control columns (§3).
"""

import numpy as np
import pandas as pd

from disputedesk.generator.comms import generate_communication_log
from disputedesk.generator.config import GeneratorConfig
from disputedesk.generator.latents import LatentBatch


def _noisy_boolean(
    rng: np.random.Generator, latent: np.ndarray, flip_prob: float, threshold: float
) -> np.ndarray:
    """Threshold(latent) + independent flip noise (defect 6, session 2). Not
    Bernoulli(latent) directly - that compounds ~30% inherent disagreement with
    the latent's own threshold before flip_prob is even applied, destroying most
    of the latent's signal (see the derivation note on the flip-prob fields in
    config.py).
    """
    raw = latent > threshold
    flip = rng.random(latent.shape[0]) < flip_prob
    return np.where(flip, ~raw, raw)


def _reason_code(
    rng: np.random.Generator, reason_subtype: np.ndarray, config: GeneratorConfig
) -> np.ndarray:
    """A noisy reading of reason_subtype (L6), not an exact copy (defect 2,
    session 2): with configured probability the recorded code is misclassified
    to a different one of the confirmed codes, modeling issuer coding error."""
    codes = list(config.reason_codes)
    code_to_idx = {code: i for i, code in enumerate(codes)}
    idx = pd.Series(reason_subtype).map(code_to_idx).to_numpy()

    n = reason_subtype.shape[0]
    misclassified = rng.random(n) < config.reason_code_misclassification_prob
    shift = rng.integers(1, len(codes), size=n)  # guarantees a genuinely different code
    recorded_idx = np.where(misclassified, (idx + shift) % len(codes), idx)
    return np.array(codes)[recorded_idx]


def _boolean_readings(
    latents: LatentBatch, rng: np.random.Generator, config: GeneratorConfig
) -> dict[str, np.ndarray]:
    threshold = config.boolean_reading_threshold
    return {
        "avs_match": _noisy_boolean(
            rng, latents.authentication_strength, config.avs_match_flip_prob, threshold
        ),
        "cvv_match": _noisy_boolean(
            rng, latents.authentication_strength, config.cvv_match_flip_prob, threshold
        ),
        "device_fingerprint_known": _noisy_boolean(
            rng, latents.relationship_genuineness, config.device_fingerprint_flip_prob, threshold
        ),
        "delivery_confirmed": _noisy_boolean(
            rng, latents.delivery_provability, config.delivery_confirmed_flip_prob, threshold
        ),
    }


def _numeric_readings(
    latents: LatentBatch, rng: np.random.Generator, config: GeneratorConfig
) -> dict[str, np.ndarray]:
    n = latents.true_fraud.shape[0]

    prior_order_count = rng.poisson(
        latents.relationship_genuineness * config.prior_order_count_scale
    )
    prior_dispute_count = rng.poisson(latents.dispute_propensity * config.prior_dispute_count_scale)

    ip_geo_base = (1.0 - latents.authentication_strength) * config.ip_geo_distance_max_km
    ip_geo_noise = rng.normal(0.0, config.ip_geo_distance_noise_km, size=n)
    ip_geo_billing_distance_km = np.clip(ip_geo_base + ip_geo_noise, 0.0, None)

    delay_noise = rng.normal(0.0, config.days_between_purchase_and_dispute_noise_days, size=n)
    days_between_purchase_and_dispute = np.clip(latents.filing_delay_days + delay_noise, 0.0, None)

    mu = np.where(
        latents.true_fraud,
        config.amount_lognormal_mu_true_fraud,
        config.amount_lognormal_mu_genuine,
    )
    amount = rng.lognormal(mean=mu, sigma=config.amount_lognormal_sigma)

    return {
        "prior_order_count": prior_order_count,
        "prior_dispute_count": prior_dispute_count,
        "ip_geo_billing_distance_km": ip_geo_billing_distance_km,
        "days_between_purchase_and_dispute": days_between_purchase_and_dispute,
        "amount": amount,
    }


def _pure_noise_readings(
    n: int, rng: np.random.Generator, config: GeneratorConfig
) -> dict[str, np.ndarray]:
    if not config.checkout_hour_uniform:
        raise NotImplementedError("only a uniform checkout_hour_of_day distribution is implemented")
    return {
        "checkout_hour_of_day": rng.integers(0, 24, size=n),
        "card_network": rng.choice(np.array(config.card_networks), size=n),
    }


def derive_features(
    latents: LatentBatch, rng: np.random.Generator, config: GeneratorConfig
) -> dict[str, np.ndarray]:
    n = latents.true_fraud.shape[0]
    comms = generate_communication_log(
        rng, latents.true_fraud, latents.relationship_genuineness, config
    )
    return {
        **_boolean_readings(latents, rng, config),
        **_numeric_readings(latents, rng, config),
        **_pure_noise_readings(n, rng, config),
        "customer_communication_log": comms,
        "reason_code": _reason_code(rng, latents.reason_subtype, config),
    }
