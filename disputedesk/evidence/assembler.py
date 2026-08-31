"""Assembles one contest evidence packet (SPEC.md §1 step 4): looks up the
required evidence types (deterministic), normalises the customer's message,
and drafts the explanation letter (both LLM, both schema-validated). This is
the only function in `evidence/` most callers need.
"""

from pydantic import BaseModel, ConfigDict

from disputedesk.evidence.context import DisputeContext
from disputedesk.evidence.draft_letter import draft_explanation_letter
from disputedesk.evidence.llm import LLMClient
from disputedesk.evidence.normalize_comms import normalize_communication_log
from disputedesk.evidence.reason_code_map import required_evidence_types
from disputedesk.evidence.schemas import ExplanationLetterOutput, NormalizedCommunicationLog


class EvidencePacket(BaseModel):
    model_config = ConfigDict(frozen=True)

    reason_code: str
    required_evidence_types: tuple[str, ...]
    normalized_comms: NormalizedCommunicationLog
    explanation_letter: ExplanationLetterOutput
    human_review_required: bool


def assemble_evidence_packet(
    context: DisputeContext, raw_communication_log: str, llm_client: LLMClient
) -> EvidencePacket:
    """Build the full evidence packet for a dispute the policy engine has
    already decided to contest. `context` and `raw_communication_log` come
    straight from the dispute row; `llm_client` is the only place this
    reaches outside `evidence/` for anything non-deterministic.
    """
    evidence_types = required_evidence_types(context.reason_code)

    comms_result = normalize_communication_log(raw_communication_log, llm_client)
    letter_result = draft_explanation_letter(
        context, evidence_types, comms_result.normalized, llm_client
    )

    return EvidencePacket(
        reason_code=context.reason_code,
        required_evidence_types=evidence_types,
        normalized_comms=comms_result.normalized,
        explanation_letter=letter_result.letter,
        human_review_required=(
            comms_result.human_review_required or letter_result.human_review_required
        ),
    )
