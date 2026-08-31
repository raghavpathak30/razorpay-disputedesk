"""GENERATOR.md §2 step 8, and the orchestration that runs every step in the exact
order §2 specifies: timestamp -> mixture component -> continuous latents -> epsilon
-> p -> features -> schema plumbing -> label. Reading top to bottom of
`generate_dataset` is reading the pipeline order.
"""

import numpy as np
import pandas as pd

from disputedesk.generator.config import GeneratorConfig
from disputedesk.generator.features import derive_features
from disputedesk.generator.latents import (
    SIMULATION_START,
    LatentBatch,
    draw_continuous_latents,
    draw_epsilon,
    draw_mixture_component,
    draw_timestamps,
)
from disputedesk.generator.probability import compute_p
from disputedesk.generator.schema import DebugRecord, DisputeRecord, assign_schema_fields


def draw_label(rng: np.random.Generator, p: np.ndarray) -> np.ndarray:
    """§2 step 8. won_if_contested ~ Bernoulli(p) - sampled, never computed."""
    return rng.random(p.shape[0]) < p


def _validate(df: pd.DataFrame, model: type) -> None:
    for row in df.to_dict(orient="records"):
        model(**row)


def _build_debug_df(
    plumbing: dict, p: np.ndarray, latents: LatentBatch, continuous: dict
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": plumbing["id"],
            "p": p,
            "true_fraud": latents.true_fraud,
            "component": latents.component,
            "period_index": latents.period_index,
            "authentication_strength": continuous["authentication_strength"],
            "relationship_genuineness": continuous["relationship_genuineness"],
            "delivery_provability": continuous["delivery_provability"],
            "filing_delay_days": continuous["filing_delay_days"],
            "dispute_propensity": continuous["dispute_propensity"],
            "reason_subtype": continuous["reason_subtype"],
            "epsilon": latents.epsilon,
        }
    )


def generate_dataset(
    n_rows: int, seed: int, config: GeneratorConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    # Step 1
    purchase_ts, period_index = draw_timestamps(rng, n_rows, config)
    # Step 2
    true_fraud, component = draw_mixture_component(rng, n_rows, period_index, config)
    # Step 3
    continuous = draw_continuous_latents(rng, n_rows, true_fraud, component, config)
    # Step 4
    epsilon = draw_epsilon(rng, n_rows, config)

    latents = LatentBatch(
        purchase_ts=purchase_ts,
        period_index=period_index,
        true_fraud=true_fraud,
        component=component,
        epsilon=epsilon,
        **continuous,
    )

    # Step 5
    p = compute_p(latents, config)
    # Step 6
    features = derive_features(latents, rng, config)
    # Step 7
    plumbing = assign_schema_fields(purchase_ts, continuous["filing_delay_days"], config)
    # Step 8
    label = draw_label(rng, p)

    features_df = pd.DataFrame({**plumbing, **features, "won_if_contested": label})
    debug_df = _build_debug_df(plumbing, p, latents, continuous)

    _validate(features_df, DisputeRecord)
    _validate(debug_df, DebugRecord)

    return features_df, debug_df


def temporal_split(
    features_df: pd.DataFrame, config: GeneratorConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split by purchase timestamp, not by row order or random assignment."""
    boundary = SIMULATION_START + pd.DateOffset(months=config.train_window_months)
    train = features_df[features_df["purchase_ts"] < boundary].copy()
    test = features_df[features_df["purchase_ts"] >= boundary].copy()
    return train, test, boundary
