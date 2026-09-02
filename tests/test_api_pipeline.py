"""Tests for `process_dispute_event` (PHASES.md Phase 4 items 2, 3, 6, 7):
the orchestration the webhook and the demo script both share. `predict_proba`
is monkeypatched to a fixed value so each test can target a specific policy
branch without training a real model - `disputedesk/model/`'s own tests
already cover the model itself.
"""

import json

import httpx
import pytest

import disputedesk.api.pipeline as pipeline_module
from disputedesk.api.pipeline import process_dispute_event
from disputedesk.api.schemas import DisputeEntity
from disputedesk.audit.db import get_engine, init_db, make_session_factory
from disputedesk.audit.log import get_decision
from disputedesk.client.razorpay import FakeRazorpayClient
from disputedesk.evidence.llm import FakeLLMClient
from disputedesk.policy.config import PolicyConfig
from disputedesk.policy.engine import Decision

POLICY = PolicyConfig(representment_cost_inr=400.0, low_confidence_band=(0.45, 0.55))
MODEL_VERSION = "test-model-v1"

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
VALID_LETTER = json.dumps(
    {
        "letter_text": "y" * 80,
        "cites_evidence_types": ["billing_proof"],
    }
)
# The grounding gate's verdict (third LLM call on the contest path, added
# 2026-09-02). Every assertion supported, so the gate passes the letter
# through unchanged and these tests keep testing what they were written to
# test - filing, idempotency, and persistence order - rather than the gate.
VALID_GROUNDING = json.dumps(
    {"assertions": [{"quote": "yy", "supporting_field": "avs_match", "verdict": "supported"}]}
)


@pytest.fixture
def session():
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    s = make_session_factory(engine)()
    yield s
    s.close()


def _entity(**overrides) -> DisputeEntity:
    fields = dict(
        id="disp_1",
        payment_id="pay_1",
        amount=5000.0,
        currency="INR",
        reason_code="MC_4837",
        phase="chargeback",
        status="open",
        avs_match=True,
        cvv_match=True,
        device_fingerprint_known=True,
        delivery_confirmed=True,
        prior_order_count=3,
        prior_dispute_count=0,
        ip_geo_billing_distance_km=10.0,
        days_between_purchase_and_dispute=3.0,
        customer_communication_log="I don't recognize this charge.",
        card_network="Mastercard",
        checkout_hour_of_day=12,
    )
    fields.update(overrides)
    return DisputeEntity(**fields)


def _mock_p_win(monkeypatch, p_win: float) -> None:
    monkeypatch.setattr(pipeline_module, "predict_proba", lambda model, X: [p_win])


def _run(session, entity, llm_client, razorpay_client, **kwargs):
    return process_dispute_event(
        entity,
        session=session,
        llm_client=llm_client,
        razorpay_client=razorpay_client,
        model=None,
        model_version=MODEL_VERSION,
        policy_config=POLICY,
        **kwargs,
    )


def test_contest_path_assembles_evidence_and_files_via_the_client(session, monkeypatch):
    _mock_p_win(monkeypatch, 0.9)  # EV = 0.9*5000-400 = 4100 > 0 -> contest
    llm = FakeLLMClient(responses=[VALID_NORMALIZED, VALID_LETTER, VALID_GROUNDING])
    razorpay = FakeRazorpayClient()

    result = _run(session, _entity(), llm, razorpay)

    assert result.policy_decision.decision == Decision.CONTEST
    assert result.decision_row.validation_result == "validated"
    assert result.decision_row.human_review_required is False
    dispute_id, amount, letter = razorpay.contest_calls[0]
    assert (dispute_id, amount) == ("disp_1", 5000.0)
    assert letter.letter_text == "y" * 80
    assert letter.submittable is True
    assert razorpay.accept_calls == []
    assert result.api_outcome.outcome == "success"


def test_accept_path_calls_accept_and_never_touches_the_llm(session, monkeypatch):
    _mock_p_win(monkeypatch, 0.01)  # EV clearly negative -> accept
    llm = FakeLLMClient(responses=["should never be called"])
    razorpay = FakeRazorpayClient()

    result = _run(session, _entity(), llm, razorpay)

    assert result.policy_decision.decision == Decision.ACCEPT
    assert result.decision_row.validation_result == "not_applicable"
    assert razorpay.accept_calls == ["disp_1"]
    assert razorpay.contest_calls == []
    assert llm.call_count == 0
    assert result.api_outcome.outcome == "success"


def test_escalate_path_never_calls_the_api(session, monkeypatch):
    _mock_p_win(monkeypatch, 0.5)  # inside the low-confidence band -> escalate
    llm = FakeLLMClient(responses=["should never be called"])
    razorpay = FakeRazorpayClient()

    result = _run(session, _entity(), llm, razorpay)

    assert result.policy_decision.decision == Decision.ESCALATE
    assert razorpay.accept_calls == []
    assert razorpay.contest_calls == []
    assert result.api_outcome is None
    assert result.decision_row.policy_branch == "escalate"


def test_decision_row_exists_before_the_api_call_is_made(session, monkeypatch):
    """PHASES.md Phase 4 item 3, proven directly: the fake client's `contest`
    queries the database mid-call, before it ever raises, and the decision
    row must already be there.
    """
    _mock_p_win(monkeypatch, 0.9)
    llm = FakeLLMClient(responses=[VALID_NORMALIZED, VALID_LETTER, VALID_GROUNDING])
    seen = {}

    class ProbeRazorpayClient:
        def accept(self, dispute_id):
            raise AssertionError("accept should not be called for this test")

        def contest(self, dispute_id, amount_inr, letter):
            seen["decision_persisted_at_call_time"] = get_decision(session, dispute_id) is not None
            raise httpx.TimeoutException("simulated - retries already exhausted inside the client")

    result = _run(session, _entity(), llm, ProbeRazorpayClient())

    assert seen["decision_persisted_at_call_time"] is True
    assert result.api_outcome.outcome == "failed"
    # The decision survives even though the API call ultimately failed -
    # SPEC.md §7 failure path 1: the system degrades, it does not lose state.
    assert get_decision(session, "disp_1") is not None


def test_replayed_event_does_not_call_the_api_a_second_time(session, monkeypatch):
    _mock_p_win(monkeypatch, 0.9)
    llm = FakeLLMClient(responses=[VALID_NORMALIZED, VALID_LETTER, VALID_GROUNDING])
    razorpay = FakeRazorpayClient()
    entity = _entity()

    result1 = _run(session, entity, llm, razorpay)
    result2 = _run(session, entity, llm, razorpay)

    assert result1.already_processed is False
    assert result2.already_processed is True
    assert len(razorpay.contest_calls) == 1
    assert razorpay.accept_calls == []
    # The replay's reported decision matches the original, from the audit row.
    assert result2.policy_decision.decision == result1.policy_decision.decision
    assert result2.policy_decision.p_win == result1.policy_decision.p_win


def test_llm_failure_degrades_to_template_and_withholds_it_from_filing(session, monkeypatch):
    """SPEC.md §7 failure path 2: the system degrades, it does not crash. What
    changed on 2026-09-02 is what "degrades" means at the filing step - the
    template letter is *not* submitted to the card network any more, because
    its own body says a person has not reviewed it (defect 0.1). The dispute
    is recorded as awaiting review instead.
    """
    _mock_p_win(monkeypatch, 0.9)
    llm = FakeLLMClient(responses=["not json", "still not json"])  # repair also fails
    razorpay = FakeRazorpayClient()

    result = _run(session, _entity(), llm, razorpay)

    assert result.decision_row.human_review_required is True
    assert result.decision_row.validation_result == "fallback_template_used"
    assert result.api_outcome.outcome == "withheld_for_review"
    assert razorpay.contest_calls == []
    assert razorpay.accept_calls == []  # never silently accepted either


def test_features_used_are_recorded_on_the_decision_row(session, monkeypatch):
    _mock_p_win(monkeypatch, 0.01)
    entity = _entity()

    result = _run(session, entity, FakeLLMClient(responses=["x"]), FakeRazorpayClient())

    features = json.loads(result.decision_row.features_json)
    assert features["amount"] == entity.amount
    assert features["reason_code"] == 0  # MC_4837 is index 0 in REASON_CODES
    assert features["card_network"] == 1  # Mastercard is index 1 in CARD_NETWORKS


def test_replayed_event_after_an_api_failure_does_not_retry_via_the_pipeline(session, monkeypatch):
    """Once a decision + failed outcome are both recorded, a replay must not
    re-attempt filing from the pipeline layer - retrying a *failed* filing is
    an operator/human decision (SPEC.md's escalate-style "I don't know" idea
    extended to a failed API call), not something a replayed webhook should
    trigger silently.
    """
    _mock_p_win(monkeypatch, 0.9)
    llm = FakeLLMClient(responses=[VALID_NORMALIZED, VALID_LETTER, VALID_GROUNDING])

    class AlwaysFailsClient:
        def __init__(self):
            self.contest_calls = []

        def accept(self, dispute_id):
            raise AssertionError

        def contest(self, dispute_id, amount_inr, letter):
            self.contest_calls.append(dispute_id)
            raise httpx.TimeoutException("simulated - retries exhausted")

    razorpay = AlwaysFailsClient()
    entity = _entity()

    result1 = _run(session, entity, llm, razorpay)
    result2 = _run(session, entity, llm, razorpay)

    assert result1.api_outcome.outcome == "failed"
    assert result2.already_processed is True
    assert result2.api_outcome.outcome == "failed"
    assert razorpay.contest_calls == ["disp_1"]  # not called again on replay
