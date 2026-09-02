"""Assembles one contest evidence packet (SPEC.md §1 step 4): looks up the
required evidence types (deterministic), normalises the customer's message,
drafts the explanation letter, and grounds it against the dispute record (all
three LLM steps schema-validated). This is the only function in `evidence/`
most callers need.

The grounding gate runs last and can only subtract: it may turn a submittable
letter into a withheld one, never the reverse
(`disputedesk/evidence/grounding.py`). `human_review_required` is read off the
letter's own provenance, so a letter the gate withheld can never be paired
with `human_review_required=False`.
"""

from pydantic import BaseModel, ConfigDict

from disputedesk.evidence.context import DisputeContext
from disputedesk.evidence.draft_letter import draft_explanation_letter
from disputedesk.evidence.grounding import GroundingVerdict, apply_grounding_gate
from disputedesk.evidence.letter import DraftedLetter
from disputedesk.evidence.llm import LLMClient
from disputedesk.evidence.normalize_comms import normalize_communication_log
from disputedesk.evidence.reason_code_map import required_evidence_types
from disputedesk.evidence.schemas import NormalizedCommunicationLog


class EvidencePacket(BaseModel):
    model_config = ConfigDict(frozen=True)

    reason_code: str
    required_evidence_types: tuple[str, ...]
    normalized_comms: NormalizedCommunicationLog
    explanation_letter: DraftedLetter
    # The gate's verdict, or None when the gate could not reach one (and the
    # letter is withheld for exactly that reason). Carried so the audit row
    # can tell "the gate withheld this letter" apart from "the gate could not
    # run" - different facts, and a reviewer needs both.
    grounding_verdict: GroundingVerdict | None
    # True if a person must read this packet before anything is filed. For
    # the letter this is not a second, separately-maintained flag: it is read
    # off the letter's own provenance, so a non-`MODEL` letter can never be
    # paired with `human_review_required=False`.
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
    letter = draft_explanation_letter(context, evidence_types, comms_result.normalized, llm_client)
    gate_result = apply_grounding_gate(letter, context, llm_client)

    return EvidencePacket(
        reason_code=context.reason_code,
        required_evidence_types=evidence_types,
        normalized_comms=comms_result.normalized,
        explanation_letter=gate_result.letter,
        grounding_verdict=gate_result.verdict,
        human_review_required=(
            comms_result.human_review_required or not gate_result.letter.submittable
        ),
    )
