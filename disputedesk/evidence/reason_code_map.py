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
    "VISA_83": _CNP_FRAUD_EVIDENCE,  # Fraud-Card Absent Environment
    "AMEX_FR2": _CNP_FRAUD_EVIDENCE,  # Fraudulent Transaction
}

assert set(REQUIRED_EVIDENCE_BY_REASON_CODE) == set(REASON_CODES)


def required_evidence_types(reason_code: str) -> tuple[str, ...]:
    """The evidence object types a contest packet for this `reason_code` must
    assemble. Raises `KeyError` on an unrecognized code - unlike the feature
    builder's ordinal encoding (which must degrade gracefully for an unseen
    category at inference time), assembling evidence for a reason code this
    system has no defense strategy for is a configuration error, not a
    runtime input to tolerate silently.
    """
    return REQUIRED_EVIDENCE_BY_REASON_CODE[reason_code]
