"""Validation behaviour for the two LLM-output schemas (SPEC.md §2, PHASES.md
Phase 3 gate: every LLM output validated against a schema before use).
"""

import pytest
from pydantic import ValidationError

from disputedesk.evidence.schemas import ExplanationLetterOutput, NormalizedCommunicationLog

VALID_NORMALIZED = {
    "claims_unauthorized_transaction": True,
    "mentions_prior_bank_contact": False,
    "mentions_shared_card_access": False,
    "mentions_travel": False,
    "tone": "polite",
    "is_substantive": True,
    "summary": "Customer states they did not authorize the charge.",
}

VALID_LETTER = {
    "letter_text": "x" * 60,
    "cites_evidence_types": ["billing_proof", "explanation_letter"],
}


def test_normalized_communication_log_accepts_a_well_formed_payload():
    model = NormalizedCommunicationLog(**VALID_NORMALIZED)
    assert model.tone == "polite"


def test_normalized_communication_log_rejects_an_invalid_tone():
    with pytest.raises(ValidationError):
        NormalizedCommunicationLog(**{**VALID_NORMALIZED, "tone": "furious"})


def test_normalized_communication_log_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        NormalizedCommunicationLog(**{**VALID_NORMALIZED, "won_if_contested": True})


def test_normalized_communication_log_rejects_an_empty_summary():
    with pytest.raises(ValidationError):
        NormalizedCommunicationLog(**{**VALID_NORMALIZED, "summary": ""})


def test_explanation_letter_output_accepts_a_well_formed_payload():
    model = ExplanationLetterOutput(**VALID_LETTER)
    assert model.cites_evidence_types == ["billing_proof", "explanation_letter"]


def test_explanation_letter_output_rejects_a_too_short_letter():
    with pytest.raises(ValidationError):
        ExplanationLetterOutput(**{**VALID_LETTER, "letter_text": "too short"})


def test_explanation_letter_output_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        ExplanationLetterOutput(**{**VALID_LETTER, "p_win": 0.9})
