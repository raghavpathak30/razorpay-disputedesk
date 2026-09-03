"""Drafting `explanation_letter` (SPEC.md §2, §3): happy path validates
directly, repeated LLM failure degrades to a deterministic template letter
(SPEC.md §7 failure path 2). Since 2026-09-02 the drafter returns a
`DraftedLetter` carrying its own `provenance` rather than a letter plus a
separate `human_review_required` boolean - the provenance *is* the review
flag, and it is what gates submission. The provenance/submission invariants
themselves are pinned in `tests/test_evidence_letter_provenance.py`.
"""

import json

from disputedesk.evidence.context import DisputeContext
from disputedesk.evidence.draft_letter import draft_explanation_letter
from disputedesk.evidence.letter import LetterProvenance
from disputedesk.evidence.llm import FakeLLMClient
from disputedesk.evidence.schemas import NormalizedCommunicationLog

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
VALID_LETTER_RESPONSE = json.dumps(
    {
        "letter_text": "x" * 80,
        "cites_evidence_types": ["billing_proof", "access_activity_log"],
    }
)


def test_valid_llm_response_is_used_directly():
    client = FakeLLMClient(responses=[VALID_LETTER_RESPONSE])
    letter = draft_explanation_letter(CONTEXT, EVIDENCE_TYPES, NORMALIZED_COMMS, client)

    assert letter.provenance is LetterProvenance.MODEL
    assert letter.letter_text == "x" * 80


def test_repair_succeeds_after_one_bad_response():
    client = FakeLLMClient(responses=["not json", VALID_LETTER_RESPONSE])
    letter = draft_explanation_letter(CONTEXT, EVIDENCE_TYPES, NORMALIZED_COMMS, client)

    assert letter.provenance is LetterProvenance.MODEL
    assert client.call_count == 2


def test_falls_back_to_deterministic_template_after_two_bad_responses():
    client = FakeLLMClient(responses=["not json", "still not json"])
    letter = draft_explanation_letter(CONTEXT, EVIDENCE_TYPES, NORMALIZED_COMMS, client)

    assert letter.provenance is LetterProvenance.FALLBACK
    assert "VISA_10_4" in letter.letter_text
    assert "5000.00" in letter.letter_text
    assert list(letter.cites_evidence_types) == list(EVIDENCE_TYPES)


def test_fallback_never_claims_the_amount_or_reason_code_are_anything_but_given():
    client = FakeLLMClient(responses=["broken", "still broken"])
    context = DisputeContext(
        reason_code="MC_4837",
        amount=999.5,
        avs_match=False,
        cvv_match=False,
        device_fingerprint_known=False,
        delivery_confirmed=False,
        prior_order_count=0,
    )
    letter = draft_explanation_letter(context, EVIDENCE_TYPES, NORMALIZED_COMMS, client)

    assert "MC_4837" in letter.letter_text
    assert "999.50" in letter.letter_text
    assert "no" in letter.letter_text
