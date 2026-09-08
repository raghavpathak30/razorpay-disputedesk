"""Unit tests for `disputedesk/evidence/grounding_baseline.py`'s new
`field_confirmations`/`numeric_confirmations` (CLAUDE.md Day-2 grounding-gate
cascade).

The case this file exists to pin: `numeric_contradictions("... 3 prior
orders ...", ctx with prior_order_count=3)` returns `[]` - and, before this
change, that `[]` was indistinguishable from a sentence with no numeric claim
at all. `numeric_confirmations` is the function that tells the two apart, and
`test_grounding_cascade.py` proves the cascade actually uses it to skip
MiniCheck.
"""

from disputedesk.evidence.context import DisputeContext
from disputedesk.evidence.grounding_baseline import (
    baseline_findings,
    baseline_flags,
    field_confirmations,
    field_contradictions,
    numeric_confirmations,
    numeric_contradictions,
)

CONTEXT = DisputeContext(
    reason_code="VISA_10_4",
    amount=1240.00,
    avs_match=True,
    cvv_match=True,
    device_fingerprint_known=True,
    delivery_confirmed=True,
    prior_order_count=3,
)


class TestNumericConfirmations:
    def test_exact_match_prior_order_count_confirms(self):
        """The exact case found while designing the cascade: an exact-match
        numeric claim must resolve as `confirmed`, not stay silent."""
        sentence = "The customer has placed 3 prior orders with us."
        findings = numeric_confirmations(sentence, CONTEXT)
        assert len(findings) == 1
        assert findings[0].kind == "confirmed"
        assert "prior_order_count=3" in findings[0].detail

    def test_exact_match_amount_confirms(self):
        sentence = "We are disputing the INR 1240.00 chargeback."
        findings = numeric_confirmations(sentence, CONTEXT)
        assert any(f.kind == "confirmed" and "amount=1240" in f.detail for f in findings)

    def test_mismatched_count_does_not_confirm(self):
        sentence = "The customer has placed 3 prior orders with us."
        mismatched = DisputeContext(**{**CONTEXT.__dict__, "prior_order_count": 7})
        assert numeric_confirmations(sentence, mismatched) == []

    def test_unrelated_sentence_does_not_confirm(self):
        """A sentence with no numeric claim at all must not be confused with
        a confirmed one - both return `[]` from `numeric_contradictions`,
        but only the exact-match case should return non-`[]` here."""
        sentence = "The weather was pleasant that day."
        assert numeric_confirmations(sentence, CONTEXT) == []


class TestNumericContradictionsUnchanged:
    """The pre-existing behaviour this change must not disturb."""

    def test_mismatched_prior_order_count_still_contradicts(self):
        sentence = "The customer has placed 3 prior orders with us."
        mismatched = DisputeContext(**{**CONTEXT.__dict__, "prior_order_count": 7})
        findings = numeric_contradictions(sentence, mismatched)
        assert len(findings) == 1
        assert findings[0].kind == "contradiction"

    def test_exact_match_still_produces_no_contradiction(self):
        sentence = "The customer has placed 3 prior orders with us."
        assert numeric_contradictions(sentence, CONTEXT) == []


class TestFieldConfirmations:
    def test_matching_boolean_polarity_confirms(self):
        sentence = "The order was delivered and confirmed at the customer's address."
        findings = field_confirmations(sentence, CONTEXT)
        assert any(f.kind == "confirmed" and "delivery_confirmed" in f.detail for f in findings)

    def test_mismatched_polarity_does_not_confirm(self):
        sentence = "The order was never delivered to the customer."
        assert field_confirmations(sentence, CONTEXT) == []

    def test_unrelated_sentence_does_not_confirm(self):
        assert field_confirmations("The weather was pleasant.", CONTEXT) == []


class TestFieldContradictionsUnchanged:
    def test_mismatched_polarity_still_contradicts(self):
        sentence = "The order was never delivered to the customer."
        findings = field_contradictions(sentence, CONTEXT)
        assert len(findings) == 1
        assert findings[0].kind == "contradiction"


class TestConfirmationsExcludedFromBaselineFlags:
    """`baseline_findings`/`baseline_flags` back the already-published
    n45/clean27 eval numbers (NUMBERS.md, README) and must not move -
    confirmations are additive for the cascade only, never folded in here."""

    def test_a_confirmed_only_sentence_does_not_flag(self):
        sentence = "The customer has placed 3 prior orders with us."
        assert baseline_findings(sentence, CONTEXT) == []
        assert baseline_flags(sentence, CONTEXT) is False
