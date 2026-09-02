"""The grounding gate: a drafted letter may not be filed until every factual
assertion in it has been traced to a field of the dispute record.

The invariants pinned here, in the order they matter:

1. **One-directional.** The gate can move a letter from submittable to
   `FAILED_GROUNDING`. It can never move a letter toward submission, and it
   never touches the policy branch.
2. **Fails closed.** Every way the grader can fail - raising, malformed JSON,
   schema violation, an invented field name - lands on `FAILED_GROUNDING`,
   never on a letter that stays submittable.
3. **Inherits the Phase 0 submit gate.** `FAILED_GROUNDING` is not `MODEL`, so
   `require_submittable` raises on it with no new code in the client.
"""

import httpx
import pytest
from pydantic import ValidationError

from disputedesk.evidence.context import DisputeContext
from disputedesk.evidence.grounding import (
    RECORD_FIELDS,
    AssertionVerdict,
    GroundingVerdict,
    apply_grounding_gate,
)
from disputedesk.evidence.letter import (
    DraftedLetter,
    LetterNotSubmittableError,
    LetterProvenance,
    require_submittable,
)
from disputedesk.evidence.llm import FakeLLMClient

CONTEXT = DisputeContext(
    reason_code="MC_4837",
    amount=4999.0,
    avs_match=True,
    cvv_match=True,
    device_fingerprint_known=True,
    delivery_confirmed=True,
    prior_order_count=6,
)

LETTER_BODY = (
    "This letter responds to a chargeback filed under reason code MC_4837 for a "
    "transaction of INR 4999.00. Address Verification Service matched the "
    "cardholder's billing address and the Card Verification Value matched. The "
    "device used at checkout was recognised from this customer's prior activity, "
    "and delivery of the order was confirmed. This customer has six prior orders "
    "on the account."
)


def _letter(provenance: LetterProvenance = LetterProvenance.MODEL) -> DraftedLetter:
    return DraftedLetter(
        letter_text=LETTER_BODY,
        cites_evidence_types=("billing_proof", "proof_of_service"),
        provenance=provenance,
    )


def _grounded_response() -> str:
    return (
        '{"assertions": ['
        '{"quote": "Address Verification Service matched", '
        '"supporting_field": "avs_match", "verdict": "supported"},'
        '{"quote": "delivery of the order was confirmed", '
        '"supporting_field": "delivery_confirmed", "verdict": "supported"}'
        "]}"
    )


def _contradicted_response() -> str:
    return (
        '{"assertions": ['
        '{"quote": "delivery of the order was confirmed", '
        '"supporting_field": "delivery_confirmed", "verdict": "contradicted"}'
        "]}"
    )


def _unsupported_response() -> str:
    return (
        '{"assertions": ['
        '{"quote": "signed for by R. Sharma", '
        '"supporting_field": null, "verdict": "unsupported"}'
        "]}"
    )


class TestProvenanceMember:
    def test_failed_grounding_is_not_submittable(self):
        letter = _letter(LetterProvenance.FAILED_GROUNDING)
        assert letter.submittable is False

    def test_failed_grounding_raises_at_the_phase_0_submit_gate(self):
        letter = _letter(LetterProvenance.FAILED_GROUNDING)
        with pytest.raises(LetterNotSubmittableError, match="failed_grounding"):
            require_submittable(letter)


class TestGatePasses:
    def test_a_fully_grounded_letter_stays_submittable(self):
        client = FakeLLMClient([_grounded_response()])
        result = apply_grounding_gate(_letter(), CONTEXT, client)
        assert result.letter.provenance is LetterProvenance.MODEL
        assert result.letter.submittable is True
        assert result.failure_reason is None
        assert result.verdict is not None and result.verdict.grounded is True

    def test_the_letter_text_is_never_rewritten_by_the_gate(self):
        client = FakeLLMClient([_grounded_response()])
        result = apply_grounding_gate(_letter(), CONTEXT, client)
        assert result.letter.letter_text == LETTER_BODY


class TestGateWithholds:
    def test_a_contradicted_assertion_withholds_the_letter(self):
        client = FakeLLMClient([_contradicted_response()])
        result = apply_grounding_gate(_letter(), CONTEXT, client)
        assert result.letter.provenance is LetterProvenance.FAILED_GROUNDING
        assert result.letter.submittable is False

    def test_an_unsupported_assertion_withholds_the_letter(self):
        """Class B: the letter asserts a fact the record has no field for."""
        client = FakeLLMClient([_unsupported_response()])
        result = apply_grounding_gate(_letter(), CONTEXT, client)
        assert result.letter.provenance is LetterProvenance.FAILED_GROUNDING

    def test_the_withheld_letter_keeps_its_text_and_citations(self):
        client = FakeLLMClient([_contradicted_response()])
        result = apply_grounding_gate(_letter(), CONTEXT, client)
        assert result.letter.letter_text == LETTER_BODY
        assert result.letter.cites_evidence_types == ("billing_proof", "proof_of_service")


class TestOneDirectional:
    @pytest.mark.parametrize(
        "provenance",
        [LetterProvenance.FALLBACK, LetterProvenance.LOW_CONFIDENCE],
    )
    def test_the_gate_never_promotes_a_non_model_letter(self, provenance):
        """Even handed a verdict saying every assertion is grounded, the gate
        cannot raise a fallback letter to submittable."""
        client = FakeLLMClient([_grounded_response()])
        result = apply_grounding_gate(_letter(provenance), CONTEXT, client)
        assert result.letter.provenance is provenance
        assert result.letter.submittable is False

    def test_the_gate_does_not_spend_a_call_on_an_already_withheld_letter(self):
        client = FakeLLMClient([_grounded_response()])
        apply_grounding_gate(_letter(LetterProvenance.FALLBACK), CONTEXT, client)
        assert client.call_count == 0


class TestFailsClosed:
    def test_a_raising_grader_withholds(self):
        class _Raising:
            def complete(self, prompt: str) -> str:
                raise httpx.ReadTimeout("grader timed out")

        result = apply_grounding_gate(_letter(), CONTEXT, _Raising())
        assert result.letter.provenance is LetterProvenance.FAILED_GROUNDING
        assert result.verdict is None
        assert "timed out" in result.failure_reason

    def test_malformed_json_twice_withholds(self):
        client = FakeLLMClient(["not json at all"])
        result = apply_grounding_gate(_letter(), CONTEXT, client)
        assert result.letter.provenance is LetterProvenance.FAILED_GROUNDING
        assert client.call_count == 2  # the one repair attempt was made

    def test_malformed_then_valid_is_repaired_and_passes(self):
        client = FakeLLMClient(["not json at all", _grounded_response()])
        result = apply_grounding_gate(_letter(), CONTEXT, client)
        assert result.letter.provenance is LetterProvenance.MODEL

    def test_a_schema_violating_verdict_withholds(self):
        client = FakeLLMClient(['{"assertions": [{"quote": "x"}]}'])
        result = apply_grounding_gate(_letter(), CONTEXT, client)
        assert result.letter.provenance is LetterProvenance.FAILED_GROUNDING

    def test_an_invented_record_field_withholds(self):
        """The grader may only cite fields that exist. A verdict naming
        `customer_loyalty_tier` is a schema violation, not a pass."""
        client = FakeLLMClient(
            [
                '{"assertions": [{"quote": "loyal customer", '
                '"supporting_field": "customer_loyalty_tier", "verdict": "supported"}]}'
            ]
        )
        result = apply_grounding_gate(_letter(), CONTEXT, client)
        assert result.letter.provenance is LetterProvenance.FAILED_GROUNDING


class TestRecordFields:
    def test_record_fields_matches_dispute_context_exactly(self):
        """The frozen allowlist and the context it describes must agree.
        Derived separately on purpose, so adding a context field cannot
        silently widen what the grader may claim support from."""
        assert RECORD_FIELDS == set(DisputeContext.__dataclass_fields__)


class TestVerdictSchema:
    def test_supporting_field_must_be_a_real_record_field(self):
        with pytest.raises(ValidationError):
            AssertionVerdict(
                quote="x", supporting_field="not_a_field", verdict="supported"
            )

    def test_every_record_field_is_accepted(self):
        for field in RECORD_FIELDS:
            AssertionVerdict(quote="x", supporting_field=field, verdict="supported")

    def test_a_supported_verdict_must_name_the_field_that_supports_it(self):
        with pytest.raises(ValidationError):
            AssertionVerdict(quote="x", supporting_field=None, verdict="supported")

    def test_an_unsupported_verdict_must_not_name_a_field(self):
        with pytest.raises(ValidationError):
            AssertionVerdict(
                quote="x", supporting_field="avs_match", verdict="unsupported"
            )

    def test_grounded_is_true_only_when_every_assertion_is_supported(self):
        supported = AssertionVerdict(
            quote="x", supporting_field="avs_match", verdict="supported"
        )
        contradicted = AssertionVerdict(
            quote="y", supporting_field="cvv_match", verdict="contradicted"
        )
        assert GroundingVerdict(assertions=[supported]).grounded is True
        assert GroundingVerdict(assertions=[supported, contradicted]).grounded is False

    def test_a_letter_with_no_extractable_assertions_is_not_grounded(self):
        """Vacuous truth is a fail-open. An empty verdict means the grader
        found nothing to check, which is not evidence the letter is clean."""
        assert GroundingVerdict(assertions=[]).grounded is False
