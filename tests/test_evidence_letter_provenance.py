"""Provenance and length invariants on the drafted `explanation_letter`.

Two defects found by the 2026-09-02 verification pass, fixed together because
they share one mechanism:

- **0.1** a deterministic fallback letter - whose own body says it has not been
  reviewed by a person - was submitted to the card network with
  `action="submit"`.
- **0.2** letters were drafted against a 4,000-character schema ceiling and
  then silently truncated to the network's real 1,000-character limit at the
  API boundary, destroying the evidence the packet exists to carry.

The invariant these tests pin: a letter object carries its own `provenance`,
assigned where it is constructed, and only `provenance == "model"` can ever
reach `action="submit"`. Everything else routes to human review, and no
network call happens on the way there.
"""

import json

import httpx
import pytest
from pydantic import ValidationError

import disputedesk.api.pipeline as pipeline_module
from disputedesk.api.pipeline import process_dispute_event
from disputedesk.api.schemas import DisputeEntity
from disputedesk.audit.db import get_engine, init_db, make_session_factory
from disputedesk.client.razorpay import FakeRazorpayClient, RazorpayHttpClient
from disputedesk.evidence.context import DisputeContext
from disputedesk.evidence.documents import EvidenceDocument
from disputedesk.evidence.draft_letter import draft_explanation_letter
from disputedesk.evidence.letter import (
    NETWORK_SUMMARY_MAX_CHARS,
    DraftedLetter,
    LetterNotSubmittableError,
    LetterProvenance,
    require_submittable,
)
from disputedesk.evidence.llm import FakeLLMClient
from disputedesk.evidence.schemas import NormalizedCommunicationLog
from disputedesk.policy.config import PolicyConfig

CONTEXT = DisputeContext(
    reason_code="VISA_10_4",
    amount=5000.0,
    avs_match=True,
    cvv_match=True,
    device_fingerprint_known=True,
    delivery_confirmed=True,
    prior_order_count=12,
)
EVIDENCE_TYPES = ("billing_proof", "access_activity_log", "explanation_letter")
NORMALIZED_COMMS = NormalizedCommunicationLog(
    claims_unauthorized_transaction=True,
    mentions_prior_bank_contact=False,
    mentions_shared_card_access=False,
    mentions_travel=False,
    tone="polite",
    is_substantive=True,
    summary="Customer says they did not authorize this charge.",
)


def _letter_response(body: str) -> str:
    return json.dumps({"letter_text": body, "cites_evidence_types": ["billing_proof"]})


def _bundle() -> tuple[EvidenceDocument, ...]:
    return (
        EvidenceDocument(evidence_type="billing_proof", filename="b.pdf", content=b"%PDF-fake"),
    )


# --------------------------------------------------------------------------
# 0.2 - the length ceiling is the network's, and it is enforced at construction
# --------------------------------------------------------------------------


def test_the_schema_ceiling_is_the_networks_real_limit():
    """The value the drafting prompt is given and the value the API accepts
    are the same number, sourced once.
    """
    assert NETWORK_SUMMARY_MAX_CHARS == 1000


def test_a_letter_over_the_network_limit_cannot_be_constructed():
    with pytest.raises(ValidationError):
        DraftedLetter(
            letter_text="x" * (NETWORK_SUMMARY_MAX_CHARS + 1),
            cites_evidence_types=("billing_proof",),
            provenance=LetterProvenance.MODEL,
        )


def test_an_over_limit_model_body_routes_to_review_and_is_never_truncated():
    """The stub returns a 3,000-character body twice (original + repair). The
    pipeline must not truncate it into a submittable letter - it degrades to
    the deterministic template, which is not submittable.
    """
    oversized = _letter_response("z" * 3000)
    client = FakeLLMClient(responses=[oversized, oversized])

    letter = draft_explanation_letter(CONTEXT, EVIDENCE_TYPES, NORMALIZED_COMMS, client)

    assert letter.provenance is not LetterProvenance.MODEL
    assert letter.submittable is False
    assert "z" * 100 not in letter.letter_text  # no truncated remnant of the model body
    with pytest.raises(LetterNotSubmittableError):
        require_submittable(letter)


# --------------------------------------------------------------------------
# 0.1 - provenance is set at construction, and gates submission
# --------------------------------------------------------------------------


def test_a_validated_model_letter_carries_model_provenance():
    client = FakeLLMClient(responses=[_letter_response("m" * 80)])
    letter = draft_explanation_letter(CONTEXT, EVIDENCE_TYPES, NORMALIZED_COMMS, client)

    assert letter.provenance is LetterProvenance.MODEL
    assert letter.submittable is True
    assert require_submittable(letter) is letter


def test_a_fallback_letter_carries_fallback_provenance():
    client = FakeLLMClient(responses=["not json", "still not json"])
    letter = draft_explanation_letter(CONTEXT, EVIDENCE_TYPES, NORMALIZED_COMMS, client)

    assert letter.provenance is LetterProvenance.FALLBACK
    assert letter.submittable is False


@pytest.mark.parametrize("provenance", [LetterProvenance.FALLBACK, LetterProvenance.LOW_CONFIDENCE])
def test_require_submittable_rejects_every_non_model_provenance(provenance):
    letter = DraftedLetter(
        letter_text="q" * 80,
        cites_evidence_types=("billing_proof",),
        provenance=provenance,
    )
    with pytest.raises(LetterNotSubmittableError):
        require_submittable(letter)


# --------------------------------------------------------------------------
# 0.1 at the type/boundary level - the client itself refuses, before the socket
# --------------------------------------------------------------------------


@pytest.fixture
def _settings_env(monkeypatch):
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_test_id")
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", "rzp_test_secret")
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_API_URL", "https://example.test/llm")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    from disputedesk.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_the_http_client_refuses_a_fallback_letter_before_any_network_call(
    monkeypatch, _settings_env
):
    def explode(*args, **kwargs):
        raise AssertionError("a non-submittable letter reached the network")

    monkeypatch.setattr(httpx, "request", explode)

    fallback = DraftedLetter(
        letter_text="this letter has not been reviewed by a person yet. " * 2,
        cites_evidence_types=("explanation_letter",),
        provenance=LetterProvenance.FALLBACK,
    )
    # A populated bundle changes nothing - 2026-09-04 reopening: provenance
    # is still checked first, so a fallback letter cannot reach the upload
    # step via this new argument either.
    with pytest.raises(LetterNotSubmittableError):
        RazorpayHttpClient().contest("disp_1", 100.0, fallback, evidence_bundle=_bundle())


def test_the_http_client_submits_a_model_letter_verbatim(monkeypatch, _settings_env):
    """A letter at the ceiling goes out whole - no `summary[:1000]` slice, and
    `action="submit"` only ever on a `model` letter.
    """
    seen = {}
    body = "m" * NETWORK_SUMMARY_MAX_CHARS

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/documents":
            return httpx.Response(200, json={"id": "doc_1"})
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "disp_1", "status": "under_review"})

    transport = httpx.MockTransport(handler)

    def fake_request(method, url, **kwargs):
        with httpx.Client(transport=transport) as http_client:
            return http_client.request(method, url, **kwargs)

    monkeypatch.setattr(httpx, "request", fake_request)

    letter = DraftedLetter(
        letter_text=body,
        cites_evidence_types=("billing_proof",),
        provenance=LetterProvenance.MODEL,
    )
    RazorpayHttpClient().contest("disp_1", 100.0, letter, evidence_bundle=_bundle())

    assert seen["body"]["summary"] == body
    assert seen["body"]["action"] == "submit"


def test_the_fake_client_enforces_the_same_invariant_as_the_real_one():
    """The demo script and most tests run against `FakeRazorpayClient`; if the
    fake were laxer than the real client, this invariant could be re-opened by
    a code path that only the demo exercises.
    """
    fallback = DraftedLetter(
        letter_text="t" * 80,
        cites_evidence_types=("explanation_letter",),
        provenance=LetterProvenance.FALLBACK,
    )
    fake = FakeRazorpayClient()
    with pytest.raises(LetterNotSubmittableError):
        fake.contest("disp_1", 100.0, fallback, evidence_bundle=_bundle())
    assert fake.contest_calls == []
    assert fake.upload_calls == []


# --------------------------------------------------------------------------
# 0.1 end to end - the pipeline withholds, and files nothing
# --------------------------------------------------------------------------

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


def test_the_pipeline_withholds_a_fallback_letter_and_files_nothing(session, monkeypatch):
    monkeypatch.setattr(pipeline_module, "predict_proba", lambda model, X: [0.9])
    llm = FakeLLMClient(responses=[VALID_NORMALIZED, "not json", "still not json"])
    razorpay = FakeRazorpayClient()

    result = process_dispute_event(
        _entity(),
        session=session,
        llm_client=llm,
        razorpay_client=razorpay,
        model=None,
        model_version="test-model-v1",
        policy_config=POLICY,
    )

    assert result.policy_decision.decision.value == "contest"
    assert razorpay.contest_calls == []
    assert razorpay.accept_calls == []
    assert result.decision_row.human_review_required is True
    assert result.api_outcome is not None
    assert result.api_outcome.outcome == "withheld_for_review"
