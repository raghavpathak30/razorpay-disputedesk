"""The reason-code -> evidence-types lookup is a deterministic table (SPEC.md
§2): every real generator reason code must resolve, and an unknown code must
fail loudly rather than silently return an empty or guessed evidence set.
"""

import pytest

from disputedesk.evidence.context import DisputeContext
from disputedesk.evidence.reason_code_map import (
    REQUIRED_EVIDENCE_BY_REASON_CODE,
    available_evidence_types,
    required_evidence_types,
)
from disputedesk.features.build import REASON_CODES


@pytest.mark.parametrize("reason_code", REASON_CODES)
def test_every_generator_reason_code_has_a_mapping(reason_code):
    evidence_types = required_evidence_types(reason_code)
    assert isinstance(evidence_types, tuple)
    assert len(evidence_types) > 0


def test_explanation_letter_is_always_required():
    # The LLM's drafted output has to actually be used by every packet, or
    # Part C's LLM boundary is dead code.
    for evidence_types in REQUIRED_EVIDENCE_BY_REASON_CODE.values():
        assert "explanation_letter" in evidence_types


def test_unknown_reason_code_raises_instead_of_guessing():
    with pytest.raises(KeyError):
        required_evidence_types("NOT_A_REAL_CODE")


def test_mapping_covers_exactly_the_generator_vocabulary():
    assert set(REQUIRED_EVIDENCE_BY_REASON_CODE) == set(REASON_CODES)


class TestAvailableEvidenceTypes:
    """2026-09-04 remediation: the drafting prompt must be told only what THIS
    dispute's own facts back up, not the reason code's full required set
    regardless of availability - see DECISIONS.md's 2026-09-04 entry."""

    REQUIRED = required_evidence_types("MC_4837")

    def test_full_signal_context_gets_the_full_required_set(self):
        context = DisputeContext(
            reason_code="MC_4837",
            amount=5000.0,
            avs_match=True,
            cvv_match=True,
            device_fingerprint_known=True,
            delivery_confirmed=True,
            prior_order_count=3,
        )
        assert available_evidence_types(context, self.REQUIRED) == self.REQUIRED

    def test_no_signal_context_gets_only_the_always_available_types(self):
        """The Dispute B / WEAK_EVIDENCE_EVENT profile: no AVS/CVV/device/
        delivery signal at all."""
        context = DisputeContext(
            reason_code="VISA_10_4",
            amount=2200.0,
            avs_match=False,
            cvv_match=False,
            device_fingerprint_known=False,
            delivery_confirmed=False,
            prior_order_count=0,
        )
        available = available_evidence_types(context, self.REQUIRED)
        assert set(available) == {"customer_communication", "explanation_letter"}

    def test_billing_proof_needs_both_avs_and_cvv_not_either_alone(self):
        context = DisputeContext(
            reason_code="MC_4837",
            amount=1000.0,
            avs_match=True,
            cvv_match=False,
            device_fingerprint_known=False,
            delivery_confirmed=False,
            prior_order_count=0,
        )
        assert "billing_proof" not in available_evidence_types(context, self.REQUIRED)

    def test_always_a_subset_of_required_never_a_superset(self):
        for reason_code in REASON_CODES:
            required = required_evidence_types(reason_code)
            for avs, cvv, device, delivery in [
                (True, True, True, True),
                (False, False, False, False),
                (True, False, True, False),
            ]:
                context = DisputeContext(
                    reason_code=reason_code,
                    amount=100.0,
                    avs_match=avs,
                    cvv_match=cvv,
                    device_fingerprint_known=device,
                    delivery_confirmed=delivery,
                    prior_order_count=0,
                )
                available = available_evidence_types(context, required)
                assert set(available) <= set(required)

    def test_result_preserves_the_required_ordering(self):
        context = DisputeContext(
            reason_code="MC_4837",
            amount=1000.0,
            avs_match=True,
            cvv_match=True,
            device_fingerprint_known=False,
            delivery_confirmed=False,
            prior_order_count=0,
        )
        available = available_evidence_types(context, self.REQUIRED)
        assert list(available) == [t for t in self.REQUIRED if t in available]
