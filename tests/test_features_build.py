"""features/ is a pure function per CLAUDE.md: dict in, dict out, hand-built
cases including near-miss confounders (a legitimate customer who looks
fraudulent, and vice versa) - no I/O, no randomness, nothing derived from the
label.
"""

import pandas as pd
import pytest

from disputedesk.features.build import (
    CARD_NETWORKS,
    FEATURE_COLUMNS,
    REASON_CODES,
    build_features,
)
from disputedesk.features.matrix import build_feature_matrix

BASE_ROW = {
    "id": "dsp_000001",
    "payment_id": "pay_000001",
    "currency": "INR",
    "phase": "chargeback",
    "status": "open",
    "purchase_ts": "2025-01-01T00:00:00",
    "filed_at": "2025-01-10T00:00:00",
    "respond_by": "2025-01-17T00:00:00",
    "avs_match": True,
    "cvv_match": False,
    "device_fingerprint_known": True,
    "delivery_confirmed": True,
    "prior_order_count": 12,
    "prior_dispute_count": 0,
    "ip_geo_billing_distance_km": 42.5,
    "days_between_purchase_and_dispute": 9.0,
    "amount": 5000.0,
    "checkout_hour_of_day": 14,
    "card_network": "Visa",
    "customer_communication_log": "I don't recognize this charge.",
    "reason_code": "VISA_10_4",
    "won_if_contested": True,
}


def test_build_features_selects_and_types_every_column():
    features = build_features(BASE_ROW)

    assert set(features) == set(FEATURE_COLUMNS)
    assert features["amount"] == 5000.0
    assert features["avs_match"] is True
    assert features["cvv_match"] is False
    assert features["prior_order_count"] == 12
    assert features["checkout_hour_of_day"] == 14
    assert features["reason_code"] == REASON_CODES.index("VISA_10_4")
    assert features["card_network"] == CARD_NETWORKS.index("Visa")


def test_build_features_ignores_the_label_and_identifiers():
    row_without_label = {k: v for k, v in BASE_ROW.items() if k != "won_if_contested"}
    # Must not raise even though the label key is absent - build_features never
    # reads it.
    features = build_features(row_without_label)
    assert "won_if_contested" not in features
    assert "id" not in features
    assert "payment_id" not in features
    assert "customer_communication_log" not in features


def test_build_features_ignores_generator_debug_fields_if_present():
    row_with_debug = {
        **BASE_ROW,
        "p": 0.91,
        "true_fraud": False,
        "component": "genuine_nonconfounder",
    }
    features = build_features(row_with_debug)
    assert "p" not in features
    assert "true_fraud" not in features
    assert "component" not in features


def test_build_features_raises_on_missing_required_key():
    incomplete_row = {k: v for k, v in BASE_ROW.items() if k != "amount"}
    with pytest.raises(KeyError):
        build_features(incomplete_row)


def test_unseen_categorical_value_maps_past_the_known_vocabulary():
    row = {**BASE_ROW, "reason_code": "RUPAY_UNKNOWN", "card_network": "Diners"}
    features = build_features(row)
    assert features["reason_code"] == len(REASON_CODES)
    assert features["card_network"] == len(CARD_NETWORKS)


# --- Near-miss confounder cases (GENERATOR.md §6) ---
# The feature builder must pass these through unchanged - it does not compute
# any fraud judgement, it only selects and encodes raw fields.


def test_traveler_confounder_features_pass_through_unchanged():
    """6A: a genuine customer (true_fraud=0) who looks risky - poor avs_match,
    high geo distance, unfamiliar device - despite a long, good order history.
    build_features must not "fix" or reinterpret this; it just encodes what's
    on the row.
    """
    traveler_row = {
        **BASE_ROW,
        "avs_match": False,
        "device_fingerprint_known": False,
        "ip_geo_billing_distance_km": 6200.0,
        "prior_order_count": 30,
    }
    features = build_features(traveler_row)
    assert features["avs_match"] is False
    assert features["device_fingerprint_known"] is False
    assert features["ip_geo_billing_distance_km"] == 6200.0
    assert features["prior_order_count"] == 30


def test_account_takeover_confounder_features_pass_through_unchanged():
    """6B: fraud (true_fraud=1) on a good account - passes auth checks cleanly
    and inherits a long order history. Looks clean on every feature even
    though it is fraud; build_features has no way to know this, by design.
    """
    takeover_row = {
        **BASE_ROW,
        "avs_match": True,
        "cvv_match": True,
        "device_fingerprint_known": True,
        "prior_order_count": 25,
        "prior_dispute_count": 0,
    }
    features = build_features(takeover_row)
    assert features["avs_match"] is True
    assert features["cvv_match"] is True
    assert features["prior_order_count"] == 25


def test_build_feature_matrix_produces_expected_columns_and_no_nans():
    df = pd.DataFrame([BASE_ROW, {**BASE_ROW, "id": "dsp_000002", "amount": 999.0}])
    matrix = build_feature_matrix(df)

    assert list(matrix.columns) == list(FEATURE_COLUMNS)
    assert not matrix.isna().any().any()
    assert len(matrix) == 2
