"""The incoming webhook's schema (PHASES.md Phase 4 item 5: "Validate the
incoming payload against a schema; a malformed webhook is rejected, not
processed").

`DisputeEntity`'s dispute-proper fields (id, payment_id, amount, currency,
reason_code, phase, status, respond_by) match SPEC.md §3 and were verified
against Razorpay's real dispute entity on 2026-09-01 -
https://razorpay.com/docs/api/disputes/fetch-all/ (see
`disputedesk/client/razorpay.py`'s module docstring for the full list of
what was verified). Two differences from Razorpay's raw wire format, both
deliberate and documented here rather than silently assumed:

1. `amount` is rupees (float), matching this codebase's one convention
   throughout (`disputedesk.generator.schema.DisputeRecord`,
   `disputedesk.features.build`, `disputedesk.policy.engine`) - not paise
   (Razorpay's own wire format). The rupees -> paise conversion happens once,
   at the `disputedesk.client.razorpay` boundary, per that module's own
   docstring.
2. `respond_by` is an ISO 8601 datetime, not a Unix timestamp, for the same
   reason.

The order-context fields (`avs_match` through `checkout_hour_of_day`) are
NOT part of Razorpay's own dispute webhook - a real deployment would look
these up from the merchant's own order/customer systems by `payment_id`
(SPEC.md §1 step 1), which this project does not build (no such lookup
exists anywhere in this codebase; out of scope). This is therefore an
ASSUMPTION about the payload shape this webhook receives, not a citation:
it is assumed to arrive already joined, shaped like
`disputedesk.generator.schema.DisputeRecord`'s own fields - the same
assumption `disputedesk/evidence/reason_code_map.py` documents for its own
uncited mapping, and for the same reason (no better source exists).

The outer envelope (`entity`, `event`, `payload.dispute.entity`,
`created_at`) follows Razorpay's well-known general webhook shape used
across its platform; no Razorpay documentation page describing this
envelope specifically for dispute events was found (two lookups 404'd) -
also an ASSUMPTION, not a citation. `status` is constrained to `"open"`
literally, since PHASES.md Phase 4 scopes this webhook to "an `open`
dispute event" specifically - this is the field this endpoint actually
gates on, not the unverified `event` string.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DisputeEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    payment_id: str
    amount: float = Field(gt=0, description="Rupees - see module docstring.")
    currency: str
    reason_code: str
    phase: Literal["fraud", "retrieval", "chargeback", "pre_arbitration", "arbitration"]
    status: Literal["open"]

    avs_match: bool
    cvv_match: bool
    device_fingerprint_known: bool
    delivery_confirmed: bool
    prior_order_count: int = Field(ge=0)
    prior_dispute_count: int = Field(ge=0)
    ip_geo_billing_distance_km: float = Field(ge=0)
    days_between_purchase_and_dispute: float = Field(ge=0)
    customer_communication_log: str
    card_network: str
    checkout_hour_of_day: int = Field(ge=0, le=23)


class DisputeEntityContainer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    entity: DisputeEntity


class DisputeWebhookPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    dispute: DisputeEntityContainer


class DisputeWebhookEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    event: str
    payload: DisputeWebhookPayload
