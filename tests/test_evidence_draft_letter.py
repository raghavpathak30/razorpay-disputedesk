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


class _RecordingLLMClient:
    """Not a `FakeLLMClient`: records every prompt it is called with, so a
    test can assert on prompt *content* rather than only on the parsed
    result."""

    def __init__(self, responses: list[str]):
        self._responses = responses
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        index = min(len(self.prompts) - 1, len(self._responses) - 1)
        return self._responses[index]


class TestPromptOnlyOffersAvailableEvidence:
    """2026-09-04 remediation: the prompt must tell the model only the
    evidence types THIS dispute's own facts back up, and name the rest as
    explicitly not being submitted - not the reason code's full required set
    regardless of availability. See DECISIONS.md's 2026-09-04 entry and
    `tests/test_evidence_reason_code_map.py::TestAvailableEvidenceTypes`."""

    WEAK_CONTEXT = DisputeContext(
        reason_code="VISA_10_4",
        amount=2200.0,
        avs_match=False,
        cvv_match=False,
        device_fingerprint_known=False,
        delivery_confirmed=False,
        prior_order_count=0,
    )
    FULL_REQUIRED = (
        "billing_proof",
        "access_activity_log",
        "proof_of_service",
        "explanation_letter",
    )

    @staticmethod
    def _submitted_and_missing_blocks(prompt: str) -> tuple[str, str]:
        """The template wraps each of these two sentences onto two lines, so
        a single-line match on the leading text would miss the values on the
        line after it. Slice on the sentences' own start markers instead -
        both are guaranteed to appear, in this order, by
        `explanation_letter_v3.txt`."""
        submitted_start = prompt.index("Evidence types actually")
        missing_start = prompt.index("Evidence types NOT")
        comms_start = prompt.index("Customer's own message")
        return prompt[submitted_start:missing_start], prompt[missing_start:comms_start]

    def test_unavailable_types_are_listed_as_missing_not_as_being_submitted(self):
        client = _RecordingLLMClient(responses=[VALID_LETTER_RESPONSE])
        draft_explanation_letter(self.WEAK_CONTEXT, self.FULL_REQUIRED, NORMALIZED_COMMS, client)

        submitted_block, missing_block = self._submitted_and_missing_blocks(client.prompts[0])
        assert "billing_proof" not in submitted_block
        assert "proof_of_service" not in submitted_block
        assert "billing_proof" in missing_block
        assert "proof_of_service" in missing_block

    def test_available_types_are_still_offered_when_the_dispute_backs_them_up(self):
        full_context = DisputeContext(
            reason_code="MC_4837",
            amount=6500.0,
            avs_match=True,
            cvv_match=True,
            device_fingerprint_known=True,
            delivery_confirmed=True,
            prior_order_count=6,
        )
        client = _RecordingLLMClient(responses=[VALID_LETTER_RESPONSE])
        draft_explanation_letter(full_context, self.FULL_REQUIRED, NORMALIZED_COMMS, client)

        submitted_block, missing_block = self._submitted_and_missing_blocks(client.prompts[0])
        assert "billing_proof" in submitted_block
        assert missing_block.splitlines()[0].strip().endswith("none")
