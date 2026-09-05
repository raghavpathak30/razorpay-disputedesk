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

import json

import httpx
import pytest
from pydantic import ValidationError

from disputedesk.evidence.context import DisputeContext
from disputedesk.evidence.grounding import (
    ALLOWED_SUPPORTING_FIELDS,
    COMMS_FIELDS,
    EVIDENCE_TYPE_FIELDS,
    RECORD_FIELDS,
    AssertionVerdict,
    GroundingVerdict,
    apply_grounding_gate,
    build_prompt,
)
from disputedesk.evidence.letter import (
    DraftedLetter,
    LetterNotSubmittableError,
    LetterProvenance,
    require_submittable,
)
from disputedesk.evidence.llm import FakeLLMClient
from disputedesk.evidence.schemas import NormalizedCommunicationLog

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
            AssertionVerdict(quote="x", supporting_field="not_a_field", verdict="supported")

    def test_every_record_field_is_accepted(self):
        for field in RECORD_FIELDS:
            AssertionVerdict(quote="x", supporting_field=field, verdict="supported")

    def test_a_supported_verdict_must_name_the_field_that_supports_it(self):
        with pytest.raises(ValidationError):
            AssertionVerdict(quote="x", supporting_field=None, verdict="supported")

    def test_an_unsupported_verdict_must_not_name_a_field(self):
        with pytest.raises(ValidationError):
            AssertionVerdict(quote="x", supporting_field="avs_match", verdict="unsupported")

    def test_grounded_is_true_only_when_every_assertion_is_supported(self):
        supported = AssertionVerdict(quote="x", supporting_field="avs_match", verdict="supported")
        contradicted = AssertionVerdict(
            quote="y", supporting_field="cvv_match", verdict="contradicted"
        )
        assert GroundingVerdict(assertions=[supported]).grounded is True
        assert GroundingVerdict(assertions=[supported, contradicted]).grounded is False

    def test_a_letter_with_no_extractable_assertions_is_not_grounded(self):
        """Vacuous truth is a fail-open. An empty verdict means the grader
        found nothing to check, which is not evidence the letter is clean."""
        assert GroundingVerdict(assertions=[]).grounded is False


class TestScopedWidening:
    """2026-09-04 remediation: `supporting_field` now also accepts an
    evidence-document type name and a `comms_*` field, so a letter that
    legitimately cites an available evidence document or the customer's own
    (normalized) message is no longer indistinguishable from one inventing a
    fact outright. `RECORD_FIELDS` itself is untouched - see
    `TestRecordFields` above, still pinned exactly as before."""

    def test_every_evidence_type_field_is_now_accepted(self):
        for field in EVIDENCE_TYPE_FIELDS:
            AssertionVerdict(quote="x", supporting_field=field, verdict="supported")

    def test_every_comms_field_is_now_accepted(self):
        for field in COMMS_FIELDS:
            AssertionVerdict(quote="x", supporting_field=field, verdict="supported")

    def test_comms_fields_are_exactly_the_normalized_communication_log_fields(self):
        assert COMMS_FIELDS == {f"comms_{name}" for name in NormalizedCommunicationLog.model_fields}

    def test_allowed_supporting_fields_is_a_strict_superset_of_record_fields(self):
        assert RECORD_FIELDS < ALLOWED_SUPPORTING_FIELDS

    def test_a_genuinely_unknown_field_is_still_rejected(self):
        """The widening adds two named surfaces; it does not accept anything
        the grader feels like inventing."""
        with pytest.raises(ValidationError):
            AssertionVerdict(
                quote="loyal customer",
                supporting_field="customer_loyalty_tier",
                verdict="supported",
            )


class TestDisputeBRegression:
    """The exact failure this remediation was checked against: a dispute with
    no AVS/CVV/device/delivery signal (the demo's WEAK_EVIDENCE_EVENT /
    "Dispute B" profile) where the model asserted a tracked delivery that
    `delivery_confirmed=False` contradicts. Widening the schema to accept
    evidence-type and comms citations must not let this specific, genuine
    fabrication back in - see DECISIONS.md's 2026-09-04 remediation entry,
    which requires this exact case to still be withheld after the fix."""

    WEAK_CONTEXT = DisputeContext(
        reason_code="VISA_10_4",
        amount=2200.0,
        avs_match=False,
        cvv_match=False,
        device_fingerprint_known=False,
        delivery_confirmed=False,
        prior_order_count=0,
    )

    def _letter_claiming_tracked_delivery(self) -> DraftedLetter:
        return DraftedLetter(
            letter_text=(
                "We contest the chargeback for INR 2200.00. Our fulfillment records "
                "confirm that the item was delivered to the billing address, as "
                "tracked by the shipping carrier."
            ),
            cites_evidence_types=("proof_of_service",),
            provenance=LetterProvenance.MODEL,
        )

    def test_the_delivery_contradiction_is_still_withheld(self):
        """The grader, informed (correctly, per the widened prompt) that
        `proof_of_service` is NOT among this dispute's submitted evidence,
        marks the claim contradicted - exactly the verdict the old, narrower
        schema already supported via `delivery_confirmed`. The gate must
        still withhold."""
        grader_response = json.dumps(
            {
                "assertions": [
                    {
                        "quote": "the item was delivered to the billing address",
                        "supporting_field": "delivery_confirmed",
                        "verdict": "contradicted",
                    }
                ]
            }
        )
        client = FakeLLMClient([grader_response])
        result = apply_grounding_gate(
            self._letter_claiming_tracked_delivery(), self.WEAK_CONTEXT, client
        )
        assert result.letter.provenance is LetterProvenance.FAILED_GROUNDING
        assert result.letter.submittable is False
        assert result.verdict is not None
        assert result.verdict.ungrounded_assertions[0].verdict == "contradicted"

    def test_citing_the_not_submitted_evidence_type_by_name_is_also_still_withheld(self):
        """Same fabrication, but the grader extracts it as a citation of the
        `proof_of_service` evidence-document type instead of the record
        field. This is now schema-legal (the widening this remediation
        added) - it must still be `contradicted`/`unsupported`, never
        `supported`, because `proof_of_service` is not in this dispute's
        available set."""
        grader_response = json.dumps(
            {
                "assertions": [
                    {
                        "quote": "as tracked by the shipping carrier",
                        "supporting_field": "proof_of_service",
                        "verdict": "contradicted",
                    }
                ]
            }
        )
        client = FakeLLMClient([grader_response])
        result = apply_grounding_gate(
            self._letter_claiming_tracked_delivery(), self.WEAK_CONTEXT, client
        )
        assert result.letter.provenance is LetterProvenance.FAILED_GROUNDING

    def test_the_prompt_tells_the_grader_proof_of_service_is_not_submitted(self):
        prompt = build_prompt(self._letter_claiming_tracked_delivery(), self.WEAK_CONTEXT)
        missing_line = next(
            line for line in prompt.splitlines() if line.startswith("Evidence documents NOT")
        )
        assert "proof_of_service" in missing_line


class TestCommsSurface:
    """The second half of the 2026-09-04 widening: a claim that paraphrases
    the customer's own (normalized) message is checkable against Section 3,
    not silently `unsupported`."""

    NORMALIZED_COMMS = NormalizedCommunicationLog(
        claims_unauthorized_transaction=True,
        mentions_prior_bank_contact=True,
        mentions_shared_card_access=False,
        mentions_travel=False,
        tone="polite",
        is_substantive=True,
        summary="Customer says they don't recognize the charge and already contacted their bank.",
    )

    def test_no_comms_record_falls_back_to_the_not_provided_sentinel(self):
        prompt = build_prompt(_letter(), CONTEXT)
        assert "not provided for this grading pass" in prompt
        assert CONTEXT.reason_code in prompt

    def test_a_real_comms_record_is_rendered_into_the_prompt(self):
        prompt = build_prompt(_letter(), CONTEXT, self.NORMALIZED_COMMS)
        assert "not provided for this grading pass" not in prompt
        assert self.NORMALIZED_COMMS.summary in prompt

    def test_a_claim_supported_by_comms_passes_the_gate(self):
        grader_response = json.dumps(
            {
                "assertions": [
                    {
                        "quote": "the customer already contacted their bank",
                        "supporting_field": "comms_mentions_prior_bank_contact",
                        "verdict": "supported",
                    }
                ]
            }
        )
        client = FakeLLMClient([grader_response])
        result = apply_grounding_gate(_letter(), CONTEXT, client, self.NORMALIZED_COMMS)
        assert result.letter.provenance is LetterProvenance.MODEL
        assert result.letter.submittable is True
