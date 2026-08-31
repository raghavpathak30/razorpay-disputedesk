"""End-to-end evidence packet assembly (SPEC.md §1 step 4): the lookup table,
the two LLM calls, and the combined human-review flag, wired together.
"""

import json

from disputedesk.evidence.assembler import assemble_evidence_packet
from disputedesk.evidence.context import DisputeContext
from disputedesk.evidence.llm import FakeLLMClient
from disputedesk.evidence.reason_code_map import required_evidence_types

CONTEXT = DisputeContext(
    reason_code="AMEX_FR2",
    amount=7500.0,
    avs_match=True,
    cvv_match=False,
    device_fingerprint_known=True,
    delivery_confirmed=True,
    prior_order_count=4,
)
VALID_NORMALIZED_RESPONSE = json.dumps(
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
VALID_LETTER_RESPONSE = json.dumps(
    {
        "letter_text": "y" * 80,
        "cites_evidence_types": ["billing_proof"],
    }
)


def test_packet_uses_the_reason_code_lookup_not_an_llm_guess():
    client = FakeLLMClient(responses=[VALID_NORMALIZED_RESPONSE, VALID_LETTER_RESPONSE])
    packet = assemble_evidence_packet(CONTEXT, "I don't recognize this charge.", client)

    assert packet.required_evidence_types == required_evidence_types(CONTEXT.reason_code)


def test_packet_is_not_flagged_for_human_review_when_both_llm_calls_succeed():
    client = FakeLLMClient(responses=[VALID_NORMALIZED_RESPONSE, VALID_LETTER_RESPONSE])
    packet = assemble_evidence_packet(CONTEXT, "I don't recognize this charge.", client)

    assert packet.human_review_required is False
    assert packet.normalized_comms.tone == "terse"
    assert packet.explanation_letter.letter_text == "y" * 80


def test_packet_is_flagged_for_human_review_when_normalization_falls_back():
    # Normalization gets two bad responses (fallback); letter drafting then
    # gets one bad, one good (repair succeeds) - the packet-level flag must
    # still be True because *one* of the two LLM jobs degraded.
    client = FakeLLMClient(responses=["broken", "still broken", "not json", VALID_LETTER_RESPONSE])
    packet = assemble_evidence_packet(CONTEXT, "please refund", client)

    assert packet.human_review_required is True
    assert packet.explanation_letter.letter_text == "y" * 80  # letter itself still succeeded


def test_packet_degrades_fully_but_does_not_crash_when_both_llm_jobs_fail():
    client = FakeLLMClient(responses=["broken"])  # every call gets this
    packet = assemble_evidence_packet(CONTEXT, "I don't recognize this charge.", client)

    assert packet.human_review_required is True
    assert "AMEX_FR2" in packet.explanation_letter.letter_text
    assert packet.normalized_comms.is_substantive is True
