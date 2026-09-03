"""GENERATOR.md §2 step 7 (schema-plumbing fields) and the two row schemas.

`DisputeRecord` is the model-facing table — every SPEC.md §3 dispute and
order-context field, plus the label. `DebugRecord` carries `p` and every latent.
The two are separate Pydantic models on purpose: nothing that constructs a
`DisputeRecord` ever has a `p` or a latent value in scope to put there.
"""

from datetime import datetime

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from disputedesk.generator.config import GeneratorConfig


class DisputeRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    payment_id: str
    amount: float
    currency: str
    reason_code: str
    phase: str
    status: str
    purchase_ts: datetime
    filed_at: datetime
    respond_by: datetime
    avs_match: bool
    cvv_match: bool
    device_fingerprint_known: bool
    delivery_confirmed: bool
    prior_order_count: int
    prior_dispute_count: int
    ip_geo_billing_distance_km: float
    days_between_purchase_and_dispute: float
    customer_communication_log: str
    card_network: str
    checkout_hour_of_day: int
    won_if_contested: bool


class DebugRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    p: float
    true_fraud: bool
    component: str
    period_index: int
    authentication_strength: float
    relationship_genuineness: float
    delivery_provability: float
    filing_delay_days: float
    dispute_propensity: float
    reason_subtype: str
    epsilon: float


LABEL_COLUMN = "won_if_contested"

# The frozen allowlist guard (a) of the leakage check asserts set equality
# against (`eval/leakage.py`). Typed out literally rather than derived from
# `DisputeRecord.model_fields`, deliberately: deriving it would mean that
# adding a latent to `DisputeRecord` silently widens the allowlist to admit it,
# which is the exact failure the guard exists to catch. The test suite asserts
# this constant and the model agree, so the two can only diverge loudly.
DISPUTE_FRAME_COLUMNS: frozenset[str] = frozenset(
    {
        "id",
        "payment_id",
        "amount",
        "currency",
        "reason_code",
        "phase",
        "status",
        "purchase_ts",
        "filed_at",
        "respond_by",
        "avs_match",
        "cvv_match",
        "device_fingerprint_known",
        "delivery_confirmed",
        "prior_order_count",
        "prior_dispute_count",
        "ip_geo_billing_distance_km",
        "days_between_purchase_and_dispute",
        "customer_communication_log",
        "card_network",
        "checkout_hour_of_day",
        LABEL_COLUMN,
    }
)

# Every column of the debug frame except `id`, which is the join key and is
# assigned from the row index alone. Any feature column whose values hash
# identically to one of these is a copy.
LATENT_FRAME_COLUMNS: frozenset[str] = frozenset(
    {
        "p",
        "true_fraud",
        "component",
        "period_index",
        "authentication_strength",
        "relationship_genuineness",
        "delivery_provability",
        "filing_delay_days",
        "dispute_propensity",
        "reason_subtype",
        "epsilon",
    }
)


def assign_schema_fields(
    purchase_ts: np.ndarray, filing_delay_days: np.ndarray, config: GeneratorConfig
) -> dict[str, np.ndarray]:
    """§2 step 7. Plumbing fields derived from the timestamp and the record index
    only - never from any causal latent used as a feature."""
    n = purchase_ts.shape[0]
    ids = np.array([f"dsp_{i:06d}" for i in range(n)])
    payment_ids = np.array([f"pay_{i:06d}" for i in range(n)])

    purchase_ts_dt = pd.to_datetime(purchase_ts)
    filed_at = purchase_ts_dt + pd.to_timedelta(filing_delay_days, unit="D")
    respond_by = filed_at + pd.to_timedelta(config.respond_by_days, unit="D")

    return {
        "id": ids,
        "payment_id": payment_ids,
        "currency": np.full(n, "INR"),
        "phase": np.full(n, "chargeback"),
        "status": np.full(n, "open"),
        "purchase_ts": purchase_ts_dt.values,
        "filed_at": filed_at.values,
        "respond_by": respond_by.values,
    }
