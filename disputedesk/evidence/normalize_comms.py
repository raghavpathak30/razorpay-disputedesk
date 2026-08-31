"""Normalises `customer_communication_log` free text into typed fields
(SPEC.md §2 - one of the LLM's exactly two allowed jobs). Degrades to a
deterministic template on repeated LLM failure (SPEC.md §7 failure path 2).
"""

from dataclasses import dataclass

from disputedesk.evidence.llm import LLMClient
from disputedesk.evidence.prompts import load_prompt
from disputedesk.evidence.schemas import NormalizedCommunicationLog
from disputedesk.evidence.validated_call import call_llm_and_validate

_MAX_SUMMARY_LENGTH = 500


@dataclass(frozen=True)
class NormalizationResult:
    normalized: NormalizedCommunicationLog
    human_review_required: bool


def _deterministic_fallback(raw_log: str) -> NormalizedCommunicationLog:
    """No LLM call. Conservative defaults only - every claim field False,
    since asserting a claim the system couldn't actually extract would be
    worse than asserting nothing. `human_review_required` on the result
    signals that a person, not this fallback, should read the real message.
    """
    text = raw_log.strip()
    summary = text if text else "(empty message)"
    if len(summary) > _MAX_SUMMARY_LENGTH:
        summary = summary[: _MAX_SUMMARY_LENGTH - 3] + "..."
    return NormalizedCommunicationLog(
        claims_unauthorized_transaction=False,
        mentions_prior_bank_contact=False,
        mentions_shared_card_access=False,
        mentions_travel=False,
        tone="neutral",
        is_substantive=len(text) > 10,
        summary=summary,
    )


def normalize_communication_log(raw_log: str, llm_client: LLMClient) -> NormalizationResult:
    prompt = load_prompt("normalize_comms_log_v1").format(raw_log=raw_log)
    parsed = call_llm_and_validate(llm_client, prompt, NormalizedCommunicationLog)
    if parsed is not None:
        return NormalizationResult(normalized=parsed, human_review_required=False)
    return NormalizationResult(
        normalized=_deterministic_fallback(raw_log), human_review_required=True
    )
