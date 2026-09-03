"""Typed schemas every LLM output is validated against before use (SPEC.md
§2, §7; PHASES.md Phase 3 gate). No LLM call in this module - schemas only.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from disputedesk.evidence.letter import LETTER_MIN_CHARS, NETWORK_SUMMARY_MAX_CHARS


class NormalizedCommunicationLog(BaseModel):
    """The typed fields SPEC.md §2 asks the LLM to normalise
    `customer_communication_log` free text into. These describe what the
    customer said, for use in assembling the evidence packet - they are not,
    and must never be treated as, a fraud signal. `policy/` never sees this
    model.
    """

    model_config = ConfigDict(extra="forbid")

    claims_unauthorized_transaction: bool = Field(
        description="Customer states they did not authorize/recognize the charge."
    )
    mentions_prior_bank_contact: bool = Field(
        description="Customer says they already contacted their bank/card issuer."
    )
    mentions_shared_card_access: bool = Field(
        description="Customer mentions a family member or other person with card access."
    )
    mentions_travel: bool = Field(
        description="Customer mentions travel as context for unfamiliar account activity."
    )
    tone: Literal["polite", "terse", "neutral"] = Field(description="Overall tone of the message.")
    is_substantive: bool = Field(
        description="False for a near-empty message ('please refund', '?', 'n/a')."
    )
    summary: str = Field(
        min_length=1,
        max_length=500,
        description="One or two sentence neutral summary of what the customer said.",
    )


class ExplanationLetterOutput(BaseModel):
    """The LLM's *raw* drafted letter, before provenance is attached. The LLM
    drafts text; it does not decide whether to contest (`policy/`'s job,
    already done before this is ever called) and states no dollar/rupee
    figures beyond `amount` as given to it.

    This is the model's output schema only. The value that travels onward is
    `disputedesk.evidence.letter.DraftedLetter`, which adds the `provenance`
    field the submission gate reads - deliberately not a field here, because
    the model must not be able to assert its own output's provenance.

    `letter_text`'s ceiling is the card network's real limit
    (`NETWORK_SUMMARY_MAX_CHARS`), not a looser drafting ceiling. Until
    2026-09-02 it was 4,000 characters and the API client truncated to 1,000
    at the wire, silently discarding three quarters of some letters; a body
    over the limit now fails validation here instead, which routes the dispute
    to the repair attempt and then to human review (SPEC.md §7 failure path 2).
    """

    model_config = ConfigDict(extra="forbid")

    letter_text: str = Field(
        min_length=LETTER_MIN_CHARS,
        max_length=NETWORK_SUMMARY_MAX_CHARS,
        description="The full explanation letter body, ready to submit as evidence.",
    )
    cites_evidence_types: list[str] = Field(
        description="Evidence types (from the required-evidence lookup) this letter refers to."
    )
