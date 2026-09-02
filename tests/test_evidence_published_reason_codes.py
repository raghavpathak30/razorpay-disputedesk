"""Every reason code Razorpay itself publishes must get through the webhook
(remediation defect 0.3).

The webhook used to constrain `reason_code` to a hand-typed four-value
`Literal`, so a dispute carrying Visa 83 - a code on Razorpay's own published
chargeback-code reference, and the exact code this project's own
`GENERATOR.md` §8 cites as its Visa source - was rejected with a 422. A 422
means "your payload is malformed"; a real code this system has no strategy for
is not a malformed payload, it is an operational gap, and the difference
matters because a 422 makes the dispute vanish rather than queue.

These tests parametrise over the committed fixture of the published list, so
they fail if the accepted set ever narrows below what Razorpay publishes.
"""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from disputedesk.api.webhook import (
    app,
    get_db_session,
    get_llm_client,
    get_model,
    get_razorpay_client,
)
from disputedesk.audit.db import get_engine, init_db, make_session_factory
from disputedesk.audit.log import get_audit_trail
from disputedesk.client.razorpay import FakeRazorpayClient
from disputedesk.evidence.llm import FakeLLMClient
from disputedesk.evidence.published_reason_codes import (
    PUBLISHED_REASON_CODES,
    is_supported_reason_code,
)
from disputedesk.evidence.reason_code_map import (
    LEGACY_WIRE_ALIASES,
    REQUIRED_EVIDENCE_BY_REASON_CODE,
    canonical_reason_code,
)

# The webhook wiring below mirrors `tests/test_api_webhook.py`'s `wired_app`.
# Duplicated rather than imported: `tests/` is not a package, so a
# cross-test-module import is import-mode dependent and would break under a
# bare `pytest` invocation in CI.

VALID_NORMALIZED = json.dumps(
    {
        "claims_unauthorized_transaction": True,
        "mentions_prior_bank_contact": False,
        "mentions_shared_card_access": False,
        "mentions_travel": False,
        "tone": "terse",
        "is_substantive": True,
        "summary": "Customer disputes the charge.",
    }
)
VALID_LETTER = json.dumps({"letter_text": "y" * 80, "cites_evidence_types": ["billing_proof"]})

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
    return {"event": "dispute.created", "payload": {"dispute": {"entity": entity}}}


class _StubModel:
    def __init__(self, p_win: float):
        self._p_win = p_win

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1 - self._p_win), np.full(n, self._p_win)])


@pytest.fixture
def wired_app():
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    fake_llm = FakeLLMClient(responses=[VALID_NORMALIZED, VALID_LETTER])
    fake_razorpay = FakeRazorpayClient()

    def _session_override():
        session = make_session_factory(engine)()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db_session] = _session_override
    app.dependency_overrides[get_llm_client] = lambda: fake_llm
    app.dependency_overrides[get_razorpay_client] = lambda: fake_razorpay
    app.dependency_overrides[get_model] = lambda: (_StubModel(0.9), "test-model-v1")

    yield TestClient(app), fake_razorpay, engine

    app.dependency_overrides.clear()


def test_the_fixture_covers_all_three_published_networks():
    networks = {c.network for c in PUBLISHED_REASON_CODES}
    assert networks == {"MASTERCARD", "VISA", "AMEX"}


def test_the_fixture_is_not_a_stub():
    """A guard against the fixture silently shrinking to the four in-scope
    codes - the exact failure mode this defect was.
    """
    assert len(PUBLISHED_REASON_CODES) >= 60


def test_wire_codes_are_unique_across_networks():
    """Mastercard and Visa both publish a code `98` and a code `99`, so a bare
    code is not an identifier - the wire form carries the network.
    """
    wire = [c.wire_code for c in PUBLISHED_REASON_CODES]
    assert len(wire) == len(set(wire))


@pytest.mark.parametrize("published", PUBLISHED_REASON_CODES, ids=lambda c: c.wire_code)
def test_every_published_reason_code_is_accepted_by_the_webhook(published, wired_app):
    client, fake_razorpay, _engine = wired_app

    response = client.post(
        "/webhooks/disputes",
        json=_event({"id": f"disp_{published.wire_code}", "reason_code": published.wire_code}),
    )

    assert response.status_code == 200, response.text


def test_a_malformed_reason_code_hits_the_fallback_path_not_a_422(wired_app):
    """Not a published code, not even a plausible one. It must still be
    accepted, tagged, and queued - never rejected at the boundary.
    """
    client, fake_razorpay, _engine = wired_app

    response = client.post(
        "/webhooks/disputes",
        json=_event({"id": "disp_malformed_rc", "reason_code": "!! not a code !!"}),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["human_review_required"] is True
    assert body["api_outcome"] == "withheld_for_review"
    assert fake_razorpay.contest_calls == []
    assert fake_razorpay.accept_calls == []


def test_an_unsupported_published_code_is_tagged_and_queued_not_filed(wired_app):
    """MC_4855 (Non-Receipt of Merchandise) is genuinely published and
    genuinely outside this project's one loss class. It is accepted, tagged,
    and left for a person - not filed, in either direction.
    """
    client, fake_razorpay, engine = wired_app

    response = client.post(
        "/webhooks/disputes",
        json=_event({"id": "disp_out_of_scope", "reason_code": "MC_4855"}),
    )

    assert response.status_code == 200
    assert fake_razorpay.contest_calls == []
    assert fake_razorpay.accept_calls == []

    session = make_session_factory(engine)()
    try:
        trail = get_audit_trail(session, "disp_out_of_scope")
    finally:
        session.close()

    assert trail.decision.validation_result == "reason_code_unrecognised"
    assert trail.decision.human_review_required is True
    assert trail.api_outcome.outcome == "withheld_for_review"


def test_visa_83_is_the_legacy_wire_form_of_the_supported_visa_condition():
    """Razorpay's published reference still lists Visa 83; Visa retired it in
    2018 under VCR and this project's value set uses the current 10.4
    condition (GENERATOR.md §8). A payload carrying the published-but-retired
    code must land on the same evidence strategy, not in the review queue.
    """
    assert LEGACY_WIRE_ALIASES["VISA_83"] == "VISA_10_4"
    assert canonical_reason_code("VISA_83") == "VISA_10_4"
    assert is_supported_reason_code("VISA_83") is True


@pytest.mark.parametrize("code", sorted(REQUIRED_EVIDENCE_BY_REASON_CODE))
def test_every_code_with_an_evidence_strategy_is_supported(code):
    assert is_supported_reason_code(code) is True


def test_an_unrecognised_code_is_not_supported():
    assert is_supported_reason_code("!! not a code !!") is False
    assert is_supported_reason_code("MC_4855") is False  # published, out of scope
