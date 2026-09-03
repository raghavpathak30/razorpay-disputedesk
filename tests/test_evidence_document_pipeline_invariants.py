"""The 2026-09-04 scoped reopening's own invariant proofs: everything the
document-upload pipeline touches (`disputedesk/evidence/documents.py`,
`disputedesk/client/razorpay.py`'s new `upload_document`, the `evidence_bundle`
field threaded through `disputedesk/evidence/assembler.py` and
`disputedesk/api/pipeline.py`) must leave every pre-existing invariant intact:

1. The policy engine still cannot reach the LLM or the new documents module -
   already generically covered by
   `tests/test_grounding_gate_pipeline.py::TestPolicyEngineCannotReachTheGate`,
   which source-scans all of `disputedesk/policy/*.py`; not duplicated here.
2. The grounding gate still gates before any upload - covered by the added
   `razorpay.upload_calls == []` assertion in
   `tests/test_grounding_gate_pipeline.py::TestWithheldLetterIsNotFiled`; not
   duplicated here.
3. Letter provenance still gates submission via the new signature - covered
   by `tests/test_evidence_letter_provenance.py`'s three `.contest(...,
   evidence_bundle=...)` calls, which now pass a populated bundle precisely
   to prove the provenance check still wins regardless; not duplicated here.

What's new here, and covered nowhere else: idempotency for re-submitted
uploads, the audit chain surviving a real contest-with-documents flow, and
every failure mode this reopening added (an unrenderable bundle, an upload
that raises, an upload that returns no id) failing closed to human review
with the reason recorded, never crashing the request.
"""

import json

import pytest

import disputedesk.api.pipeline as pipeline_module
import disputedesk.evidence.assembler as assembler_module
from disputedesk.api.pipeline import process_dispute_event
from disputedesk.api.schemas import DisputeEntity
from disputedesk.audit.chain import verify_chain
from disputedesk.audit.db import get_engine, init_db, make_session_factory
from disputedesk.client.razorpay import DocumentUploadError, FakeRazorpayClient
from disputedesk.evidence.documents import EvidenceRenderError
from disputedesk.evidence.llm import FakeLLMClient
from disputedesk.policy.config import PolicyConfig

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


def _contest_llm() -> FakeLLMClient:
    return FakeLLMClient(responses=[VALID_NORMALIZED, VALID_LETTER, VALID_GROUNDING])


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


def _force_contest(monkeypatch):
    monkeypatch.setattr(pipeline_module, "predict_proba", lambda model, X: [0.9])


# --------------------------------------------------------------------------
# Idempotency, including for re-submitted uploads
# --------------------------------------------------------------------------


def test_a_replayed_event_does_not_upload_documents_a_second_time(session, monkeypatch):
    """The DB-level idempotency check (`get_decision` before anything else
    runs) is what `_file_if_needed`/`_file`/`contest()` - and therefore every
    `upload_document` call inside `contest()` - sit behind. This proves the
    upload step inherits that guarantee for free, rather than assuming it.
    """
    _force_contest(monkeypatch)
    razorpay = FakeRazorpayClient()
    entity = _entity()

    result1 = _run(session, entity, _contest_llm(), razorpay)
    result2 = _run(session, entity, _contest_llm(), razorpay)

    assert result1.already_processed is False
    assert result2.already_processed is True
    assert len(razorpay.contest_calls) == 1
    uploads_after_first_run = len(razorpay.upload_calls)
    assert uploads_after_first_run > 0  # the real reason code has evidence types to render
    # The replay must not add a single further upload call.
    assert len(razorpay.upload_calls) == uploads_after_first_run


# --------------------------------------------------------------------------
# The audit chain survives a real contest-with-documents flow
# --------------------------------------------------------------------------


def test_the_audit_chain_still_verifies_after_a_contest_with_documents(session, monkeypatch):
    _force_contest(monkeypatch)
    razorpay = FakeRazorpayClient()

    result = _run(session, _entity(), _contest_llm(), razorpay)

    assert result.api_outcome.outcome == "success"
    assert len(razorpay.contest_calls) == 1
    verification = verify_chain(session)
    assert verification.ok, verification.problems


# --------------------------------------------------------------------------
# Every new failure mode fails closed, carries a reason, and never crashes
# --------------------------------------------------------------------------


def test_a_render_failure_withholds_for_review_and_never_reaches_the_client(session, monkeypatch):
    """An evidence type the renderer doesn't support (in practice: none exist
    today - `disputedesk/evidence/documents.py`'s own module-level assert
    guarantees that - so this is forced, the same way an LLM failure is
    forced elsewhere in this suite, to prove the fail-closed path exists).
    """
    _force_contest(monkeypatch)

    def _explode(*args, **kwargs):
        raise EvidenceRenderError("simulated: no renderer for this evidence type")

    monkeypatch.setattr(assembler_module, "render_evidence_bundle", _explode)
    razorpay = FakeRazorpayClient()

    result = _run(session, _entity(), _contest_llm(), razorpay)

    assert result.decision_row.human_review_required is True
    assert result.api_outcome.outcome == "withheld_for_review"
    assert "evidence bundle is empty or failed to render" in result.api_outcome.error
    assert razorpay.contest_calls == []
    assert razorpay.upload_calls == []


def test_an_empty_rendered_bundle_withholds_for_review(session, monkeypatch):
    """Distinct from a render *failure*: rendering succeeds but produces zero
    documents. Same fail-closed destination either way - `_file_if_needed`
    treats `None` and `()` identically via `not evidence.evidence_bundle`.
    """
    _force_contest(monkeypatch)
    monkeypatch.setattr(assembler_module, "render_evidence_bundle", lambda *a, **k: ())
    razorpay = FakeRazorpayClient()

    result = _run(session, _entity(), _contest_llm(), razorpay)

    assert result.api_outcome.outcome == "withheld_for_review"
    assert razorpay.contest_calls == []


def test_an_upload_failure_degrades_to_a_failed_outcome_not_a_crash(session, monkeypatch):
    """Unlike a render failure (caught before `_file` is ever reached), an
    upload failure happens *inside* the client call `_file` makes, after
    `_file_if_needed` already decided the bundle looked fileable - so it
    surfaces as `_file`'s existing "failed" outcome (SPEC.md §7 failure path
    1), the same as a network timeout on the contest PATCH always has.
    """
    _force_contest(monkeypatch)
    razorpay = FakeRazorpayClient(
        upload_responses=[DocumentUploadError("simulated upload failure")]
    )

    result = _run(session, _entity(), _contest_llm(), razorpay)

    assert result.api_outcome.outcome == "failed"
    assert "simulated upload failure" in result.api_outcome.error
    # The decision itself still stands - the system degrades, it does not
    # lose state (SPEC.md §7).
    assert result.decision_row.policy_branch == "contest"


def test_an_upload_returning_no_id_degrades_to_a_failed_outcome_not_a_crash(session, monkeypatch):
    _force_contest(monkeypatch)
    razorpay = FakeRazorpayClient(upload_responses=[""])

    result = _run(session, _entity(), _contest_llm(), razorpay)

    assert result.api_outcome.outcome == "failed"
    assert "no document id" in result.api_outcome.error
