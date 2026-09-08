"""Drafts the `explanation_letter` evidence object (SPEC.md §2 - the LLM's
other allowed job, and SPEC.md §3's `explanation_letter` evidence type).
Degrades to a deterministic template on repeated LLM failure (SPEC.md §7
failure path 2). Takes only the dispute's order-context fields and the
already-normalised communication log - never decides contest/accept
(`policy/` already decided that before this is ever called), never computes
money beyond quoting the given `amount`.

Every return from this module is a `DraftedLetter` carrying its own
`provenance`, assigned on the branch that built it: the validated-model
branch is the only one that produces `LetterProvenance.MODEL`, and only that
value can ever be submitted (`disputedesk/evidence/letter.py`). There is no
longer a separate `human_review_required` boolean travelling beside the
letter - review is implied by `provenance is not MODEL`, so the two cannot
disagree.
"""

from disputedesk.evidence.context import DisputeContext
from disputedesk.evidence.letter import (
    LETTER_MIN_CHARS,
    NETWORK_SUMMARY_MAX_CHARS,
    DraftedLetter,
    LetterProvenance,
    fallback_text_and_provenance,
)
from disputedesk.evidence.llm import LLMClient
from disputedesk.evidence.prompts import load_prompt
from disputedesk.evidence.reason_code_map import available_evidence_types
from disputedesk.evidence.schemas import ExplanationLetterOutput, NormalizedCommunicationLog
from disputedesk.evidence.validated_call import call_llm_and_validate

PROMPT_VERSION = "explanation_letter_v4"
"""v4 (2026-09-07) is a byte-ordering change only, not a content/meaning
change: split into `explanation_letter_v4_static.txt` (role, invention/
citation rules, hard limit, JSON schema - zero per-dispute variance) and
`explanation_letter_v4_dynamic.txt` (dispute record, evidence lists, comms
fields - always last), so Groq's exact-prefix prompt caching can hit on the
static portion across every call regardless of which dispute is being
drafted. See `_STATIC_PREFIX` below. Versioned as a new file rather than an
edit in place, per `disputedesk/evidence/prompts.py`, even though nothing the
model is told has changed - old audit rows' `prompt_version` must keep
pointing at the exact bytes they actually ran with.

v3 (2026-09-04) tells the model only the evidence types THIS dispute's own
facts actually back up (`reason_code_map.available_evidence_types`), lists the
rest as explicitly NOT being submitted, and instructs the model to state that
gap rather than invent supporting narrative for it. v2 passed the reason
code's full required set regardless of per-dispute availability, which was
read by the model as "these are all being submitted" and produced exactly the
fabricated claims (a delivery confirmation, an access-log match) the
grounding gate then had to withhold - see DECISIONS.md's 2026-09-04
remediation entry. v2 (2026-09-02) states the card network's character limit
in the prompt itself; v1 quoted a 4,000-character budget the wire format
could not carry."""

_STATIC_PREFIX = load_prompt(f"{PROMPT_VERSION}_static").format(
    min_chars=LETTER_MIN_CHARS, max_chars=NETWORK_SUMMARY_MAX_CHARS
)
"""Computed once at import time, so every call in this process shares the
exact same object - the module-level constant Groq's exact-prefix caching
needs (CLAUDE.md Day-1 Phase 1: "one changed byte early in the prompt is a
full miss"). `min_chars`/`max_chars` are baked in here rather than left as
per-call `.format()` slots because, while they are technically read from
`disputedesk.evidence.letter`, their *values* never vary across calls - so
formatting them once at import keeps this prefix provably byte-identical
without duplicating the two constants as literals in the prompt file
(`disputedesk/evidence/letter.py`'s own docstring is why that single-source
rule exists). `tests/test_evidence_draft_letter.py`'s
`test_two_prompt_builds_for_different_disputes_share_a_byte_identical_prefix`
pins this."""

_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "explanation_letter_output",
        "schema": ExplanationLetterOutput.model_json_schema(),
        "strict": True,
    },
}
"""CLAUDE.md Day-1 Phase 2: constrains the drafting call's output at the
sampling layer via Groq's structured outputs, so a malformed/non-schema
response is no longer a failure mode this call needs a repair retry for.
Derived from `ExplanationLetterOutput.model_json_schema()` directly, once at
import time - no second, hand-written source of truth for the letter's shape.
`ExplanationLetterOutput`'s `extra="forbid"` is what makes this schema valid
for Groq's `strict: true` mode (which requires `additionalProperties: false`
at every object level); nothing here re-derives or overrides that.

Verified live against `openai/gpt-oss-20b` before this constant was written
(this session's throwaway, uncommitted smoke script): HTTP 200, schema
accepted, `finish_reason: "stop"`, output parsed and validated against
`ExplanationLetterOutput` with zero repairs. Only this call passes
`response_format`/`repair=False` to `call_llm_and_validate` -
`normalize_comms.py` and `grounding.py` are unaffected, by construction: they
never pass either argument."""


def _deterministic_fallback(
    context: DisputeContext, evidence_types: tuple[str, ...]
) -> DraftedLetter:
    """No LLM call. States only the raw order-context facts, with no
    narrative framing - a person reviews and improves it later, which the
    returned letter's non-`MODEL` provenance is what actually enforces.
    """
    sentences = [
        f"This letter responds to a chargeback filed under reason code {context.reason_code} "
        f"for a transaction of INR {context.amount:.2f}.",
        f"Address Verification Service match: {'yes' if context.avs_match else 'no'}.",
        f"Card Verification Value match: {'yes' if context.cvv_match else 'no'}.",
        "Device recognized from this customer's prior activity: "
        f"{'yes' if context.device_fingerprint_known else 'no'}.",
        f"Delivery or fulfillment of this transaction: "
        f"{'confirmed' if context.delivery_confirmed else 'not confirmed'}.",
        f"This customer has {context.prior_order_count} prior order(s) on this account.",
        "This letter was generated by a deterministic template because "
        "automated drafting was unavailable; it has not been reviewed by a person yet.",
    ]
    text, provenance = fallback_text_and_provenance(" ".join(sentences))
    return DraftedLetter(
        letter_text=text,
        cites_evidence_types=tuple(evidence_types),
        provenance=provenance,
    )


def draft_explanation_letter(
    context: DisputeContext,
    evidence_types: tuple[str, ...],
    normalized_comms: NormalizedCommunicationLog,
    llm_client: LLMClient,
) -> DraftedLetter:
    available = available_evidence_types(context, evidence_types)
    missing = tuple(t for t in evidence_types if t not in available)
    dynamic_record = load_prompt(f"{PROMPT_VERSION}_dynamic").format(
        reason_code=context.reason_code,
        amount=f"{context.amount:.2f}",
        avs_match=context.avs_match,
        cvv_match=context.cvv_match,
        device_fingerprint_known=context.device_fingerprint_known,
        delivery_confirmed=context.delivery_confirmed,
        prior_order_count=context.prior_order_count,
        evidence_types=", ".join(available),
        missing_evidence_types=", ".join(missing) if missing else "none",
        comms_summary=normalized_comms.summary,
        comms_tone=normalized_comms.tone,
    )
    # `_STATIC_PREFIX` first, dynamic dispute record last (CLAUDE.md Day-1
    # Phase 1) - every byte before this concatenation point is identical
    # across every call this process makes, regardless of dispute.
    prompt = _STATIC_PREFIX + "\n\n" + dynamic_record
    parsed = call_llm_and_validate(
        llm_client, prompt, ExplanationLetterOutput, response_format=_RESPONSE_FORMAT, repair=False
    )
    if parsed is not None:
        # The only construction site of a submittable letter in this codebase.
        # `parsed` is the model's own output, already validated against
        # `ExplanationLetterOutput` - including the network length limit, so
        # nothing is shortened here or anywhere downstream.
        return DraftedLetter(
            letter_text=parsed.letter_text,
            cites_evidence_types=tuple(parsed.cites_evidence_types),
            provenance=LetterProvenance.MODEL,
        )

    return _deterministic_fallback(context, evidence_types)
