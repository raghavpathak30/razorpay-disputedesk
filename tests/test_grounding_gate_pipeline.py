"""The grounding gate inside the full pipeline: what a withheld letter does to
filing, to the audit row, and to the policy branch.

The three things pinned here that the unit tests cannot reach:

1. A letter the gate withholds is **not filed**, in either direction, and no
   network call is made on the way to that outcome.
2. The **policy branch on the audit row is unchanged**. The gate is not a
   second decision-maker: `policy_branch` still reads `contest`, and what
   changed is that the packet was not fit to file - a separate fact, recorded
   separately, exactly as the reason-code and drafting-failure paths already
   do.
3. The gate **cannot cause a filing**. It has no path to `accept`, no path to
   `contest`, and cannot convert an ESCALATE into anything.
"""

import json

import pytest

import disputedesk.api.pipeline as pipeline_module
from disputedesk.api.pipeline import process_dispute_event
from disputedesk.api.schemas import DisputeEntity
from disputedesk.audit.db import get_engine, init_db, make_session_factory
from disputedesk.client.razorpay import FakeRazorpayClient
from disputedesk.evidence.letter import LetterProvenance
from disputedesk.evidence.llm import FakeLLMClient
from disputedesk.policy.config import PolicyConfig
from disputedesk.policy.engine import Decision

POLICY = PolicyConfig(representment_cost_inr=400.0, low_confidence_band=(0.45, 0.55))

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
GROUNDED = json.dumps(
    {"assertions": [{"quote": "yy", "supporting_field": "avs_match", "verdict": "supported"}]}
)
UNGROUNDED = json.dumps(
    {
        "assertions": [
            {"quote": "signed for by R. Sharma", "supporting_field": None, "verdict": "unsupported"}
        ]
    }
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
        id="disp_g1",
        payment_id="pay_g1",
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


def _run(session, entity, llm, razorpay):
    return process_dispute_event(
        entity,
        session=session,
        llm_client=llm,
        razorpay_client=razorpay,
        model=None,
        model_version="test-model-v1",
        policy_config=POLICY,
    )


def _contest(monkeypatch):
    monkeypatch.setattr(pipeline_module, "predict_proba", lambda model, X: [0.9])


class TestWithheldLetterIsNotFiled:
    def test_nothing_is_filed_in_either_direction(self, session, monkeypatch):
        _contest(monkeypatch)
        razorpay = FakeRazorpayClient()
        llm = FakeLLMClient([VALID_NORMALIZED, VALID_LETTER, UNGROUNDED])

        result = _run(session, _entity(), llm, razorpay)

        assert razorpay.contest_calls == []
        assert razorpay.accept_calls == []
        # 2026-09-04 reopening: a withheld letter must not reach the upload
        # step either - the gate still gates before any document is filed,
        # not just before the contest PATCH.
        assert razorpay.upload_calls == []
        assert result.api_outcome.outcome == "withheld_for_review"
        assert "failed_grounding" in result.api_outcome.error

    def test_the_audit_row_records_the_gate_as_the_reason(self, session, monkeypatch):
        _contest(monkeypatch)
        llm = FakeLLMClient([VALID_NORMALIZED, VALID_LETTER, UNGROUNDED])

        result = _run(session, _entity(), llm, FakeRazorpayClient())

        assert result.decision_row.validation_result == "grounding_gate_withheld"
        assert result.decision_row.human_review_required is True

    def test_a_gate_failure_is_distinguishable_from_a_drafting_failure(self, session, monkeypatch):
        """`grounding_gate_withheld` and `fallback_template_used` are different
        rows. The first means a model letter exists and says something the
        record does not support; the second means there is no model letter at
        all. A reviewer needs to know which."""
        _contest(monkeypatch)
        drafting_failed = FakeLLMClient([VALID_NORMALIZED, "broken"])
        result = _run(session, _entity(id="disp_g2"), drafting_failed, FakeRazorpayClient())
        assert result.decision_row.validation_result == "fallback_template_used"


class TestThePolicyEngineKeepsVeto:
    def test_the_policy_branch_is_unchanged_by_a_gate_withhold(self, session, monkeypatch):
        _contest(monkeypatch)
        llm = FakeLLMClient([VALID_NORMALIZED, VALID_LETTER, UNGROUNDED])

        result = _run(session, _entity(), llm, FakeRazorpayClient())

        assert result.policy_decision.decision is Decision.CONTEST
        assert result.decision_row.policy_branch == "contest"

    def test_a_withheld_dispute_is_never_silently_accepted(self, session, monkeypatch):
        """Accepting is irreversible. A gate failure is not evidence about
        whether the dispute is winnable, so it must not be converted into an
        accept - the same argument `_withhold_for_review` already makes for
        the drafting-failure and unknown-reason-code paths."""
        _contest(monkeypatch)
        razorpay = FakeRazorpayClient()
        llm = FakeLLMClient([VALID_NORMALIZED, VALID_LETTER, UNGROUNDED])

        _run(session, _entity(), llm, razorpay)

        assert razorpay.accept_calls == []

    def test_the_gate_never_runs_on_a_non_contest_branch(self, session, monkeypatch):
        """ESCALATE: no evidence is assembled at all, so the gate costs
        nothing and has nothing to act on. It cannot turn an escalation into
        a filing."""
        monkeypatch.setattr(pipeline_module, "predict_proba", lambda model, X: [0.5])
        llm = FakeLLMClient(["should never be called"])
        razorpay = FakeRazorpayClient()

        result = _run(session, _entity(), llm, razorpay)

        assert result.policy_decision.decision is Decision.ESCALATE
        assert llm.call_count == 0
        assert razorpay.contest_calls == [] and razorpay.accept_calls == []


class TestGroundedLetterStillFiles:
    def test_a_grounded_letter_is_filed_exactly_as_before(self, session, monkeypatch):
        _contest(monkeypatch)
        razorpay = FakeRazorpayClient()
        llm = FakeLLMClient([VALID_NORMALIZED, VALID_LETTER, GROUNDED])

        result = _run(session, _entity(), llm, razorpay)

        assert result.decision_row.validation_result == "validated"
        dispute_id, amount, letter, evidence_bundle = razorpay.contest_calls[0]
        assert (dispute_id, amount) == ("disp_g1", 5000.0)
        assert letter.provenance is LetterProvenance.MODEL
        assert len(evidence_bundle) > 0
        assert result.api_outcome.outcome == "success"


class TestPolicyEngineCannotReachTheGate:
    def test_the_policy_package_imports_nothing_from_evidence(self):
        """CLAUDE.md invariant 4, re-asserted for the new module: adding an
        LLM surface must not create a path from `policy/` to `evidence/`."""
        import pathlib

        for path in pathlib.Path("disputedesk/policy").glob("*.py"):
            source = path.read_text()
            assert "disputedesk.evidence" not in source, path
            assert "grounding" not in source, path
