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
