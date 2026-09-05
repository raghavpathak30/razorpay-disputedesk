"""Reason code -> required evidence types. A deterministic lookup table only
(SPEC.md §2: "An LLM here is strictly worse and will be marked down.") - no
LLM anywhere in this module.

`GENERATOR.md` §8 sources the four `reason_code` values themselves from
Razorpay's published chargeback reason-code reference
(`https://cdn.razorpay.com/files/chargeback_codes.pdf`), confirming they are
all card-not-present fraud-category codes. That source lists each code's
network and reason text but not a *per-code evidence checklist* - card
networks publish evidence-matrix documents separately, and none was fetched
for this project. The mapping below is therefore an ASSUMPTION, not a
citation: all four codes are CNP "the cardholder didn't authorize this"
fraud claims (see `GENERATOR.md` §8's table), so this project defends each of
them with the same practical evidence set a merchant would actually have on
hand - proof the transaction was authenticated, proof it was genuinely
fulfilled, and the customer's own communication - rather than inventing
network-specific requirements with no source behind them.
"""

from disputedesk.evidence.context import DisputeContext
from disputedesk.features.build import REASON_CODES

# SPEC.md §3's evidence object types, restricted to what a CNP fraud dispute
# can plausibly need. `others` and `refund_cancellation_policy` /
# `terms_and_conditions` / `cancellation_proof` / `refund_confirmation` are
# real SPEC.md §3 evidence types but don't apply to an unauthorized-charge
# claim (they answer "was this order cancelled/refunded already", a
# different reason-code family) - left out deliberately, not by omission.
_CNP_FRAUD_EVIDENCE: tuple[str, ...] = (
    "billing_proof",  # AVS/CVV match: the charge was authenticated as this cardholder
    "access_activity_log",  # device fingerprint / IP-geo: this was the genuine account
    "proof_of_service",  # delivery/fulfillment: the transaction was real, not phantom
    "customer_communication",  # the raw log itself, filed as evidence
    "explanation_letter",  # LLM-drafted narrative tying the above together
)

REQUIRED_EVIDENCE_BY_REASON_CODE: dict[str, tuple[str, ...]] = {
    "MC_4837": _CNP_FRAUD_EVIDENCE,  # No Cardholder Authorization
    "MC_4840": _CNP_FRAUD_EVIDENCE,  # Fraudulent Processing of Transactions
    "VISA_10_4": _CNP_FRAUD_EVIDENCE,  # Other Fraud - Card-Absent (supersedes legacy Visa 83)
    "AMEX_FR2": _CNP_FRAUD_EVIDENCE,  # Fraudulent Transaction
}

assert set(REQUIRED_EVIDENCE_BY_REASON_CODE) == set(REASON_CODES)

LEGACY_WIRE_ALIASES: dict[str, str] = {
    # Visa retired standalone reason code 83 in 2018 under Visa Claims
    # Resolution, reclassifying card-absent fraud into the 10.x conditions;
    # 10.4 "Other Fraud - Card-Absent Environment" is the current equivalent
    # and is the key used above. Razorpay's published reference has not been
    # updated and still lists 83 (re-checked 2026-09-02), so a payload sourced
    # from that reference will carry the retired code for a dispute this
    # system *does* handle. Mapping it forward is not a guess: it is the same
    # dispute condition under its current name. See GENERATOR.md §8 and
    # DECISIONS.md's 2026-09-01 "Visa reason code rename" entry.
    "VISA_83": "VISA_10_4",
}


def canonical_reason_code(reason_code: str) -> str:
    """Map a retired published code onto the current condition this system
    keys on. Every other code is returned unchanged - this is a rename table,
    not a normaliser, and it must never guess.
    """
    return LEGACY_WIRE_ALIASES.get(reason_code, reason_code)


def required_evidence_types(reason_code: str) -> tuple[str, ...]:
    """The evidence object types a contest packet for this `reason_code` must
    assemble. Raises `KeyError` on an unrecognized code - unlike the feature
    builder's ordinal encoding (which must degrade gracefully for an unseen
    category at inference time), assembling evidence for a reason code this
    system has no defense strategy for is a configuration error, not a
    runtime input to tolerate silently.

    Callers reached here only after `is_supported_reason_code`
    (`disputedesk/evidence/published_reason_codes.py`) said yes - since
    2026-09-02 the webhook no longer rejects an unknown code at the boundary,
    so this `KeyError` is a genuine last-resort guard rather than the thing
    standing between an unknown code and a crash.
    """
    return REQUIRED_EVIDENCE_BY_REASON_CODE[canonical_reason_code(reason_code)]


# `customer_communication` and `explanation_letter` never depend on a
# per-dispute signal: the raw communication log is always present on a
# dispute row, and the explanation letter is always drafted - by the LLM, or
# by the deterministic fallback if that fails
# (`disputedesk/evidence/draft_letter.py`).
_ALWAYS_AVAILABLE_EVIDENCE: frozenset[str] = frozenset(
    {"customer_communication", "explanation_letter"}
)


def available_evidence_types(context: DisputeContext, required: tuple[str, ...]) -> tuple[str, ...]:
    """Which of `required`'s evidence types THIS dispute's own order-context
    facts actually back up - a strict subset of `required`, never a superset.

    `required_evidence_types` says what a reason code needs in general;
    it says nothing about whether a *given* dispute can actually supply it.
    Feeding the full required set to the drafting prompt regardless of
    availability told the model evidence was "being submitted" that in fact
    was not, and the model filled the gap by inventing supporting narrative
    for it (2026-09-04 remediation - see DECISIONS.md). This function is the
    one place that per-dispute judgment is made, so the drafting prompt
    (`draft_letter.py`) and any future caller share the same answer instead
    of each re-deriving it.

    Deterministic, no I/O, no LLM (SPEC.md §2) - mirrors exactly the
    evidence-document renderers' own preconditions in
    `disputedesk/evidence/documents.py` (billing_proof needs both AVS and
    CVV to match; access_activity_log needs a known device; proof_of_service
    needs confirmed delivery), so "available to cite in the letter" and
    "renderable as a document" cannot silently diverge.
    """
    available = set(_ALWAYS_AVAILABLE_EVIDENCE)
    if context.avs_match and context.cvv_match:
        available.add("billing_proof")
    if context.device_fingerprint_known:
        available.add("access_activity_log")
    if context.delivery_confirmed:
        available.add("proof_of_service")
    return tuple(t for t in required if t in available)
