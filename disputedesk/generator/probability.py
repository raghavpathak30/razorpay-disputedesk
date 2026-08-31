"""GENERATOR.md §2 step 5: compute p from the latents drawn in latents.py, and map
it into the configured band. p is a logistic function of L1-L7 plus the irreducible
residual epsilon (§1, §4) — nothing here reads a feature or the label; both are
downstream of this module in the pipeline.

Calibration note (documented here, not just in config.py, so the arithmetic behind
the coefficients survives review): at mean latent values, `logit_intercept` +
sum(coef * mean_latent) is chosen so that the true_fraud=0 case rescales to
~p_mode_high_target and true_fraud=1 (which additionally subtracts coef_true_fraud)
rescales to ~p_mode_low_target, per GENERATOR.md §4's stated modes (0.08 / 0.39).
This is a first-order calibration against a two-point idealization, exactly like
GENERATOR.md §5's own worked illustration — the realized empirical modes are
checked against these targets after generation, not assumed to match exactly.
"""

import numpy as np
import pandas as pd

from disputedesk.generator.config import GeneratorConfig
from disputedesk.generator.latents import LatentBatch


def _reason_subtype_offsets(config: GeneratorConfig) -> dict[str, float]:
    codes = config.reason_codes
    spread = np.linspace(-1.0, 1.0, len(codes)) * config.reason_subtype_logit_offset_scale
    return dict(zip(codes, spread, strict=True))


def compute_p(latents: LatentBatch, config: GeneratorConfig) -> np.ndarray:
    normalized_delay = np.minimum(
        latents.filing_delay_days / config.filing_delay_norm_cap_days, 1.0
    )
    reason_offset = (
        pd.Series(latents.reason_subtype).map(_reason_subtype_offsets(config)).to_numpy()
    )

    logit = (
        config.logit_intercept
        - config.coef_true_fraud * latents.true_fraud.astype(float)
        + config.coef_authentication_strength * latents.authentication_strength
        + config.coef_relationship_genuineness * latents.relationship_genuineness
        + config.coef_delivery_provability * latents.delivery_provability
        + config.coef_filing_delay_norm * normalized_delay
        + config.coef_dispute_propensity * latents.dispute_propensity
        + reason_offset
        + latents.epsilon
    )
    sigmoid = 1.0 / (1.0 + np.exp(-logit))
    p = config.p_min + (config.p_max - config.p_min) * sigmoid
    return np.clip(p, config.p_min, config.p_max)
