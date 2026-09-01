"""Tests for the FastAPI webhook route itself (PHASES.md Phase 4 item 1, 5):
HTTP-level behaviour, with the DB, LLM, and Razorpay client all overridden
via `app.dependency_overrides` to fakes (CLAUDE.md: "No test may make a
network call."). `get_default_model_bundle` (a real ~15k-row LightGBM
train) is never exercised here - `get_model` is overridden too, so these
tests stay fast; `disputedesk/model/`'s own tests cover the model.
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
from disputedesk.client.razorpay import FakeRazorpayClient
from disputedesk.evidence.llm import FakeLLMClient

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
    """`predict_proba` returns a fixed P(win) - what branch fires is
    controlled by the test, not by an actually-trained model.
    """

    def __init__(self, p_win: float):
        self._p_win = p_win

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1 - self._p_win), np.full(n, self._p_win)])


@pytest.fixture
def wired_app(monkeypatch):
    """Wires the app to an isolated in-memory DB and fake LLM/Razorpay
    clients, and yields (`TestClient`, `FakeRazorpayClient`, `engine`) so a
    test can both call the route and inspect what was recorded.
    """
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


def test_valid_open_dispute_event_returns_200_and_contests(wired_app):
    client, fake_razorpay, _engine = wired_app

    response = client.post("/webhooks/disputes", json=_event())

    assert response.status_code == 200
    body = response.json()
    assert body["dispute_id"] == "disp_1"
    assert body["decision"] == "contest"
    assert body["already_processed"] is False
    assert fake_razorpay.contest_calls == [("disp_1", 5000.0, "y" * 80)]


def test_malformed_webhook_is_rejected_with_422_and_not_processed(wired_app):
    client, fake_razorpay, _engine = wired_app

    response = client.post("/webhooks/disputes", json=_event({"status": "won"}))

    assert response.status_code == 422
    assert fake_razorpay.contest_calls == []
    assert fake_razorpay.accept_calls == []


def test_missing_field_is_rejected_with_422(wired_app):
    client, _fake_razorpay, _engine = wired_app
    entity = {k: v for k, v in VALID_ENTITY.items() if k != "amount"}

    response = client.post(
        "/webhooks/disputes",
        json={"event": "dispute.created", "payload": {"dispute": {"entity": entity}}},
    )

    assert response.status_code == 422


def test_replayed_webhook_event_does_not_double_file(wired_app):
    client, fake_razorpay, _engine = wired_app
    event = _event()

    first = client.post("/webhooks/disputes", json=event)
    second = client.post("/webhooks/disputes", json=event)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["already_processed"] is False
    assert second.json()["already_processed"] is True
    assert len(fake_razorpay.contest_calls) == 1


def test_llm_degradation_still_returns_200_and_flags_human_review(wired_app):
    client, fake_razorpay, _engine = wired_app
    app.dependency_overrides[get_llm_client] = lambda: FakeLLMClient(
        responses=["not json", "still not json"]
    )

    response = client.post("/webhooks/disputes", json=_event({"id": "disp_degraded"}))

    assert response.status_code == 200
    assert response.json()["human_review_required"] is True
    assert fake_razorpay.contest_calls[0][0] == "disp_degraded"  # still filed, just degraded
