"""Tests for `disputedesk/evidence/grounding_cascade.py` (CLAUDE.md Day-2
grounding-gate cascade). No network, no real model load - `minicheck.py`'s
`FakeMiniCheckClient`/`NeverCallMiniCheckClient` stand in throughout.
"""

from disputedesk.evidence.context import DisputeContext
from disputedesk.evidence.grounding_cascade import (
    MINICHECK_THRESHOLD,
    apply_grounding_cascade,
    grade_letter_with_cascade,
)
from disputedesk.evidence.letter import DraftedLetter, LetterProvenance
from disputedesk.evidence.minicheck import FakeMiniCheckClient, NeverCallMiniCheckClient

CONTEXT = DisputeContext(
    reason_code="VISA_10_4",
    amount=1240.00,
    avs_match=True,
    cvv_match=True,
    device_fingerprint_known=True,
    delivery_confirmed=True,
    prior_order_count=3,
)


def _letter(text: str, provenance: LetterProvenance = LetterProvenance.MODEL) -> DraftedLetter:
    return DraftedLetter(
        letter_text=text,
        cites_evidence_types=("billing_proof",),
        provenance=provenance,
    )


class TestConfirmedSentenceSkipsMiniCheck:
    """The exact case DECISIONS.md's 2026-09-08 entry is about: an
    exact-match numeric claim must resolve at stage 1, and MiniCheck - the
    stage that scored this very sentence a coin-flip 0.49 in the Phase 1
    smoke test - must never be consulted for it.
    """

    def test_three_prior_orders_resolves_at_stage_1_and_minicheck_is_never_called(self):
        letter_text = "The customer has placed 3 prior orders with us. " * 3  # pad to min length
        minicheck = NeverCallMiniCheckClient()

        result = apply_grounding_cascade(_letter(letter_text), CONTEXT, minicheck)

        assert result.letter.provenance is LetterProvenance.MODEL
        assert result.letter.submittable is True
        assert result.verdict is not None
        assert all(v.stage == "baseline_confirmed" for v in result.verdict.sentence_verdicts)

    def test_same_case_with_a_call_counting_spy_shows_zero_calls(self):
        """Belt and suspenders on the same proof, using a call-count spy
        instead of a raising one, in case the raising spy ever masks a
        different code path incidentally not calling `.score()`."""
        letter_text = "The customer has placed 3 prior orders with us. " * 3
        minicheck = FakeMiniCheckClient([0.99])

        apply_grounding_cascade(_letter(letter_text), CONTEXT, minicheck)

        assert minicheck.call_count == 0


class TestContradictionWithholdsAndSkipsMiniCheck:
    def test_mismatched_prior_order_count_withholds_without_calling_minicheck(self):
        mismatched = DisputeContext(**{**CONTEXT.__dict__, "prior_order_count": 7})
        letter_text = "The customer has placed 3 prior orders with us. " * 3
        minicheck = NeverCallMiniCheckClient()

        result = apply_grounding_cascade(_letter(letter_text), mismatched, minicheck)

        assert result.letter.provenance is LetterProvenance.FAILED_GROUNDING
        assert "baseline_contradiction" in result.failure_reason


class TestUnresolvedSentenceCallsMiniCheck:
    def test_a_sentence_with_no_baseline_signal_reaches_minicheck(self):
        letter_text = "Our records show this transaction was entirely legitimate. " * 3
        minicheck = FakeMiniCheckClient([0.95])

        result = apply_grounding_cascade(_letter(letter_text), CONTEXT, minicheck)

        assert minicheck.call_count >= 1
        assert result.letter.provenance is LetterProvenance.MODEL

    def test_a_low_minicheck_score_withholds(self):
        letter_text = "Our records show this transaction was entirely legitimate. " * 3
        minicheck = FakeMiniCheckClient([0.1])

        result = apply_grounding_cascade(_letter(letter_text), CONTEXT, minicheck)

        assert result.letter.provenance is LetterProvenance.FAILED_GROUNDING
        assert "minicheck" in result.failure_reason

    def test_a_score_exactly_at_threshold_passes(self):
        """Fails closed on 'below', not 'at' - the threshold itself is a
        pass, matching `score >= MINICHECK_THRESHOLD`."""
        letter_text = "Our records show this transaction was entirely legitimate. " * 3
        minicheck = FakeMiniCheckClient([MINICHECK_THRESHOLD])

        verdict = grade_letter_with_cascade(letter_text, CONTEXT, minicheck)

        assert verdict.grounded is True


class TestFailsClosed:
    def test_minicheck_unavailable_withholds_rather_than_passing(self):
        class _AlwaysUnavailable:
            def score(self, document: str, claim: str) -> float:
                from disputedesk.evidence.minicheck import MiniCheckUnavailableError

                raise MiniCheckUnavailableError("model not loaded")

        letter_text = "Our records show this transaction was entirely legitimate. " * 3
        result = apply_grounding_cascade(_letter(letter_text), CONTEXT, _AlwaysUnavailable())

        assert result.letter.provenance is LetterProvenance.FAILED_GROUNDING
        assert result.verdict.minicheck_failure is None  # no score to report, only unavailability

    def test_minicheck_failure_reports_the_lowest_scoring_sentence(self):
        letter_text = (
            "Our records show this transaction was entirely legitimate. "
            "The account in question has a long and trustworthy history. "
        )
        minicheck = FakeMiniCheckClient([0.6, 0.2])

        verdict = grade_letter_with_cascade(letter_text, CONTEXT, minicheck)

        failure = verdict.minicheck_failure
        assert failure is not None
        assert failure.minicheck_score == 0.2


class TestOneDirectional:
    def test_a_non_model_letter_is_returned_untouched(self):
        minicheck = NeverCallMiniCheckClient()
        letter = _letter(
            "This is a fallback letter that has not been reviewed. " * 2,
            LetterProvenance.FALLBACK,
        )

        result = apply_grounding_cascade(letter, CONTEXT, minicheck)

        assert result.letter is letter
        assert result.verdict is None
