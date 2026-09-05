"""Assembles one contest evidence packet (SPEC.md §1 step 4): looks up the
required evidence types (deterministic), normalises the customer's message,
drafts the explanation letter, grounds it against the dispute record (all
three LLM steps schema-validated), then renders the evidence bundle those
same required types describe into uploadable documents. This is the only
function in `evidence/` most callers need.

The grounding gate runs before rendering and can only subtract: it may turn a
submittable letter into a withheld one, never the reverse
(`disputedesk/evidence/grounding.py`). `human_review_required` is read off the
letter's own provenance and off whether a bundle could be rendered at all, so
neither a withheld letter nor a missing bundle can be paired with
`human_review_required=False`.

**2026-09-04 scoped reopening:** `evidence_bundle` is new. Rendering is pure
(`disputedesk.evidence.documents.render_evidence_bundle` - no I/O, no
network) and runs after the gate, so a withheld letter's text can still be
rendered into an (unsubmittable, review-only) explanation-letter document;
what actually blocks a submit is `client.razorpay.contest()`'s own
`require_submittable` check, not anything here. A render failure
(`EvidenceRenderError` - an unknown or unsupported evidence type) fails
closed: `evidence_bundle` is `None`, and `human_review_required` becomes
`True`, the same fail-closed shape a withheld letter already gets.
"""

from pydantic import BaseModel, ConfigDict

from disputedesk.evidence.context import DisputeContext
from disputedesk.evidence.documents import (
    EvidenceDocument,
    EvidenceRenderError,
    render_evidence_bundle,
)
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
    # The rendered, uploadable evidence documents, or None if rendering
    # failed (an unsupported evidence type - see EvidenceRenderError). Not
    # otherwise gated here: whether a document set may actually be submitted
    # is client.razorpay.contest()'s call, not this module's.
    evidence_bundle: tuple[EvidenceDocument, ...] | None
    # True if a person must read this packet before anything is filed. For
    # the letter this is not a second, separately-maintained flag: it is read
    # off the letter's own provenance, so a non-`MODEL` letter can never be
    # paired with `human_review_required=False`. Same for the bundle: a
    # render failure forces this True regardless of the letter's provenance.
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
    gate_result = apply_grounding_gate(letter, context, llm_client, comms_result.normalized)

    try:
        evidence_bundle = render_evidence_bundle(
            context,
            comms_result.normalized,
            raw_communication_log,
            gate_result.letter,
            evidence_types,
        )
    except EvidenceRenderError:
        evidence_bundle = None

    return EvidencePacket(
        reason_code=context.reason_code,
        required_evidence_types=evidence_types,
        normalized_comms=comms_result.normalized,
        explanation_letter=gate_result.letter,
        grounding_verdict=gate_result.verdict,
        evidence_bundle=evidence_bundle,
        human_review_required=(
            comms_result.human_review_required
            or not gate_result.letter.submittable
            or evidence_bundle is None
        ),
    )
