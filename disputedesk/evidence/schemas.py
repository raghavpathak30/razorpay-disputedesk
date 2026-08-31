"""Typed schemas every LLM output is validated against before use (SPEC.md
§2, §7; PHASES.md Phase 3 gate). No LLM call in this module - schemas only.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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
    """The `explanation_letter` evidence object's drafted content (SPEC.md
    §3's `explanation_letter` evidence type). The LLM drafts text; it does
    not decide whether to contest (`policy/`'s job, already done before this
    is ever called) and states no dollar/rupee figures beyond `amount` as
    given to it.
    """

    model_config = ConfigDict(extra="forbid")

    letter_text: str = Field(
        min_length=50,
        max_length=4000,
        description="The full explanation letter body, ready to submit as evidence.",
    )
    cites_evidence_types: list[str] = Field(
        description="Evidence types (from the required-evidence lookup) this letter refers to."
    )
