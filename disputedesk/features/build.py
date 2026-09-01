"""Pure feature-building functions. No I/O, no network, no randomness.

Turns one dispute/order-context row (a dict with the same keys as
`disputedesk.generator.schema.DisputeRecord`, or a real webhook payload shaped the
same way) into the flat, numeric feature dict the model consumes. Every function
here is dict in, dict out - testable with hand-built cases, per CLAUDE.md.

Excluded on purpose, and why:
- `id`, `payment_id`: identifiers, not predictive signal.
- `currency`, `phase`, `status`: constant schema-plumbing fields (GENERATOR.md
  §8) - zero variance, nothing for a model to learn.
- `purchase_ts`, `filed_at`, `respond_by`: raw timestamps. The timing signal
  that matters (GENERATOR.md §1 L5) is already captured by the noisy
  `days_between_purchase_and_dispute` reading; the timestamps themselves are
  schema plumbing, not a modeled cause.
- `customer_communication_log`: free text. Normalising it into structured
  fields is `evidence/`'s job starting in Phase 3 (SPEC.md §2), not a raw
  LightGBM tabular input - nothing outside `evidence/` may treat this text as
  a feature source.
- `won_if_contested`: the label.
"""

REASON_CODES: tuple[str, ...] = ("MC_4837", "MC_4840", "VISA_10_4", "AMEX_FR2")
CARD_NETWORKS: tuple[str, ...] = ("Visa", "Mastercard", "RuPay", "Amex")

# Column order the model is trained and predicted on. Kept as one list so
# training and inference can never silently disagree on column order.
FEATURE_COLUMNS: tuple[str, ...] = (
    "amount",
    "avs_match",
    "cvv_match",
    "device_fingerprint_known",
    "delivery_confirmed",
    "prior_order_count",
    "prior_dispute_count",
    "ip_geo_billing_distance_km",
    "days_between_purchase_and_dispute",
    "checkout_hour_of_day",
    "reason_code",
    "card_network",
)

# Passed to LightGBM's `categorical_feature` - encoded as fixed ordinals below,
# never fit on data, so training and inference use the same mapping.
CATEGORICAL_FEATURE_COLUMNS: tuple[str, ...] = ("reason_code", "card_network")


def _encode_categorical(value: str, vocabulary: tuple[str, ...]) -> int:
    """Ordinal-encode against a fixed, known vocabulary (GENERATOR.md §8's
    confirmed reason codes and card networks). Never fit on data - an unseen
    value is mapped past the end of the vocabulary rather than raising, so a
    real webhook payload with a code this dataset never generated still
    produces a valid feature row.
    """
    try:
        return vocabulary.index(value)
    except ValueError:
        return len(vocabulary)


def build_features(row: dict) -> dict:
    """Select and encode the model-facing feature columns from one dispute row.

    Reads only the keys it needs; ignores everything else in `row` (including a
    `won_if_contested` label or generator-debug fields like `p`, if present) so
    it can be called on both a clean webhook payload and a generator output row.
    """
    return {
        "amount": float(row["amount"]),
        "avs_match": bool(row["avs_match"]),
        "cvv_match": bool(row["cvv_match"]),
        "device_fingerprint_known": bool(row["device_fingerprint_known"]),
        "delivery_confirmed": bool(row["delivery_confirmed"]),
        "prior_order_count": int(row["prior_order_count"]),
        "prior_dispute_count": int(row["prior_dispute_count"]),
        "ip_geo_billing_distance_km": float(row["ip_geo_billing_distance_km"]),
        "days_between_purchase_and_dispute": float(row["days_between_purchase_and_dispute"]),
        "checkout_hour_of_day": int(row["checkout_hour_of_day"]),
        "reason_code": _encode_categorical(row["reason_code"], REASON_CODES),
        "card_network": _encode_categorical(row["card_network"], CARD_NETWORKS),
    }
