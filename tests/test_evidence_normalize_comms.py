"""Normalising `customer_communication_log` (SPEC.md §2): the happy path
validates directly, and repeated LLM failure degrades to the deterministic
template with `human_review_required=True` (SPEC.md §7 failure path 2).
"""

import json

from disputedesk.evidence.llm import FakeLLMClient
from disputedesk.evidence.normalize_comms import normalize_communication_log

VALID_RESPONSE = json.dumps(
    {
        "claims_unauthorized_transaction": True,
        "mentions_prior_bank_contact": True,
        "mentions_shared_card_access": False,
        "mentions_travel": False,
        "tone": "polite",
        "is_substantive": True,
        "summary": "Customer says they did not authorize the charge and already called their bank.",
    }
)


def test_valid_llm_response_is_used_directly():
    client = FakeLLMClient(responses=[VALID_RESPONSE])
    result = normalize_communication_log("I don't recognize this charge.", client)

    assert result.human_review_required is False
    assert result.normalized.claims_unauthorized_transaction is True
    assert result.normalized.tone == "polite"


def test_one_normalization_costs_exactly_one_call_when_the_first_response_validates():
    # A live-call check made against this function separately called
    # `.complete()` directly and then this function again, doubling the API
    # requests for one logical normalisation - this pins the fix.
    client = FakeLLMClient(responses=[VALID_RESPONSE])
    normalize_communication_log("I don't recognize this charge.", client)

    assert client.call_count == 1


def test_repair_succeeds_after_one_bad_response():
    client = FakeLLMClient(responses=["not json", VALID_RESPONSE])
    result = normalize_communication_log("I don't recognize this charge.", client)

    assert result.human_review_required is False
    assert client.call_count == 2


def test_falls_back_to_deterministic_template_after_two_bad_responses():
    client = FakeLLMClient(responses=["not json", "still not json"])
    result = normalize_communication_log("I don't recognize this charge.", client)

    assert result.human_review_required is True
    assert result.normalized.claims_unauthorized_transaction is False
    assert result.normalized.tone == "neutral"
    assert "recognize" in result.normalized.summary


def test_fallback_never_calls_the_llm_for_content_it_cannot_extract():
    # Conservative defaults, not an invented claim - the fallback must not
    # assert claims_unauthorized_transaction=True just because that's a
    # common case; it has no way to know that without the LLM.
    client = FakeLLMClient(responses=["broken", "still broken"])
    result = normalize_communication_log("?", client)

    assert result.normalized.claims_unauthorized_transaction is False
    assert result.normalized.mentions_prior_bank_contact is False
    assert result.normalized.is_substantive is False


def test_fallback_summary_is_truncated_for_a_very_long_message():
    client = FakeLLMClient(responses=["broken", "still broken"])
    long_message = "a" * 1000
    result = normalize_communication_log(long_message, client)

    assert len(result.normalized.summary) <= 500
