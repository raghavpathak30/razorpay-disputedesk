"""Tests for the incoming webhook schema (PHASES.md Phase 4 item 5: "a
malformed webhook is rejected, not processed").
"""

import pytest
from pydantic import ValidationError

from disputedesk.api.schemas import DisputeWebhookEvent

VALID_ENTITY = {
    "id": "disp_1",
    "payment_id": "pay_1",
    "amount": 5000.0,
    "currency": "INR",
    "reason_code": "MC_4837",
    "phase": "chargeback",
    "status": "open",
    "avs_match": True,
    "cvv_match": True,
    "device_fingerprint_known": True,
    "delivery_confirmed": True,
    "prior_order_count": 3,
    "prior_dispute_count": 0,
    "ip_geo_billing_distance_km": 10.0,
    "days_between_purchase_and_dispute": 3.0,
    "customer_communication_log": "I don't recognize this charge.",
    "card_network": "Mastercard",
    "checkout_hour_of_day": 12,
}


def _event(entity_overrides: dict | None = None) -> dict:
    entity = {**VALID_ENTITY, **(entity_overrides or {})}
    return {
        "event": "dispute.created",
        "payload": {"dispute": {"entity": entity}},
    }


def test_a_valid_open_dispute_event_parses():
    event = DisputeWebhookEvent.model_validate(_event())

    assert event.payload.dispute.entity.id == "disp_1"
    assert event.payload.dispute.entity.status == "open"


def test_status_other_than_open_is_rejected():
    with pytest.raises(ValidationError):
        DisputeWebhookEvent.model_validate(_event({"status": "won"}))


def test_missing_required_field_is_rejected():
    entity = {**VALID_ENTITY}
    del entity["reason_code"]
    with pytest.raises(ValidationError):
        DisputeWebhookEvent.model_validate(
            {"event": "dispute.created", "payload": {"dispute": {"entity": entity}}}
        )


def test_wrong_type_for_a_field_is_rejected():
    with pytest.raises(ValidationError):
        DisputeWebhookEvent.model_validate(_event({"amount": "not a number"}))


def test_negative_amount_is_rejected():
    with pytest.raises(ValidationError):
        DisputeWebhookEvent.model_validate(_event({"amount": -100.0}))


def test_invalid_phase_is_rejected():
    with pytest.raises(ValidationError):
        DisputeWebhookEvent.model_validate(_event({"phase": "not_a_real_phase"}))


def test_checkout_hour_out_of_range_is_rejected():
    with pytest.raises(ValidationError):
        DisputeWebhookEvent.model_validate(_event({"checkout_hour_of_day": 24}))


def test_missing_envelope_payload_is_rejected():
    with pytest.raises(ValidationError):
        DisputeWebhookEvent.model_validate({"event": "dispute.created"})


@pytest.mark.parametrize("reason_code", ["NOT_A_REAL_CODE", "VISA_83", "MC_4855", "!! junk !!"])
def test_an_unrecognized_reason_code_is_accepted_at_the_boundary(reason_code):
    """Reversed on 2026-09-02 (defect 0.3). This test previously asserted the
    opposite - that anything outside the four codes with an evidence strategy
    was rejected here - which meant Visa 83, a code on Razorpay's own
    published reference, was 422'd at the boundary with no audit row and no
    queue entry.

    Shape validation belongs here; "do we have a strategy for this code" does
    not, and the two must not share an outcome. The routing of an unrecognised
    code to the documented fallback (tagged `reason_code_unrecognised`, queued,
    nothing filed) is covered end to end in
    `tests/test_evidence_published_reason_codes.py`.
    """
    event = DisputeWebhookEvent.model_validate(_event({"reason_code": reason_code}))

    assert event.payload.dispute.entity.reason_code == reason_code


@pytest.mark.parametrize("bad_reason_code", ["", "x" * 65])
def test_a_reason_code_outside_the_length_bound_is_still_rejected(bad_reason_code):
    """The one constraint that remains is a cost/DoS bound, not a
    known-values check.
    """
    with pytest.raises(ValidationError):
        DisputeWebhookEvent.model_validate(_event({"reason_code": bad_reason_code}))


@pytest.mark.parametrize("bad_id", ["disp/../secrets", "disp?x=1", "", "x" * 65])
def test_id_with_unsafe_or_out_of_range_characters_is_rejected(bad_id):
    # `id` is interpolated directly into the Razorpay API request path
    # (disputedesk/client/razorpay.py) - a `/` or `?` here could alter the
    # request target, not just misname the dispute.
    with pytest.raises(ValidationError):
        DisputeWebhookEvent.model_validate(_event({"id": bad_id}))


def test_communication_log_over_the_length_cap_is_rejected():
    with pytest.raises(ValidationError):
        DisputeWebhookEvent.model_validate(_event({"customer_communication_log": "x" * 5001}))


def test_extra_unknown_fields_on_the_entity_are_ignored_not_rejected():
    # Real Razorpay dispute entities carry an `evidence` sub-object this
    # webhook doesn't need - unknown fields shouldn't make an otherwise
    # valid payload "malformed".
    event = DisputeWebhookEvent.model_validate(
        _event({"evidence": {"summary": None}, "amount_deducted": 0})
    )
    assert event.payload.dispute.entity.id == "disp_1"
