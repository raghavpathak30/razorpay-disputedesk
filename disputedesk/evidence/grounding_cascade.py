"""Grounding-gate cascade, stage 1 (deterministic) + stage 2 (MiniCheck),
CLAUDE.md Day-2 grounding-gate cascade.

**A new path, not a replacement.** `disputedesk/evidence/grounding.py`'s
Groq-based `apply_grounding_gate` is untouched and still the production
gate - Day 6 needs it intact as the comparison baseline. This module is not
called from `assembler.py`/`api/pipeline.py` yet; it exists to be tested on
its own before any decision is made about wiring it into the live path. See
DECISIONS.md's 2026-09-08 entry.

**Per-sentence, not per-letter.** Stage 1
(`disputedesk/evidence/grounding_baseline.py`) and stage 2
(`disputedesk/evidence/minicheck.py`) both operate on the same sentence list,
produced once by `grounding_baseline.sentences()` - so "one sentence" means
the same thing to both stages, and MiniCheck never sees the Groq gate's own
assertion extraction (the whole point of a stage 2 that does not depend on a
Groq call).

**Per sentence, in order:**

1. A field or numeric **contradiction** (`field_contradictions`/
   `numeric_contradictions`) withholds immediately. Strongest, cheapest
   signal; stage 2 is not consulted.
2. Failing that, an **unrecorded-entity shape** (`unrecorded_entities`,
   Class B - tracking numbers, signatures, phone/email contact, dates,
   IP/login, named people) withholds immediately. This is exactly what the
   baseline is *not* good at in general, but a shape hit is still a
   deterministic, explainable reason to withhold - stage 2 is not consulted.
3. Failing that, a **confirmed** field or numeric match
   (`field_confirmations`/`numeric_confirmations`) resolves the sentence as
   grounded and skips stage 2 entirely. See DECISIONS.md's 2026-09-08 entry
   for why this exists: without it, every non-contradicted sentence -
   including ones stage 1 could already resolve with full confidence - fell
   through to MiniCheck, making it the sole authority on exactly the claim
   class (exact numeric/boolean matches) a deterministic checker is
   naturally best at.
4. Otherwise the sentence is genuinely unresolved by stage 1, and MiniCheck
   scores it. **Fails closed**: a score below `MINICHECK_THRESHOLD`, or a
   `MiniCheckUnavailableError` from stage 2 itself, withholds - "unclear" is
   never treated as "pass".
"""

from dataclasses import dataclass
from typing import Literal

from disputedesk.evidence.context import DisputeContext
from disputedesk.evidence.grounding_baseline import (
    field_confirmations,
    field_contradictions,
    numeric_confirmations,
    numeric_contradictions,
    sentences,
    unrecorded_entities,
)
from disputedesk.evidence.letter import DraftedLetter, LetterProvenance
from disputedesk.evidence.minicheck import MiniCheckScorer, MiniCheckUnavailableError

MINICHECK_THRESHOLD = 0.7
"""PROVISIONAL (CLAUDE.md Day-2 Phase 2) - picked conservatively (bias
toward withholding), not tuned against a corpus. Phase 1's six hand-written
sentences scored confidently-grounded at 0.82-0.88 and
confidently-ungrounded at 0.03-0.17 on this model/hardware; 0.7 sits with
real margin below the grounded cluster and well above the ungrounded one.
Exact-match numeric/boolean claims - the case that scored a coin-flip 0.49
in that same Phase 1 run - no longer reach this threshold at all, because
stage 1's `*_confirmations` functions resolve them first. Day 5 is the
scheduled tuning pass; this number is not that pass."""

SentenceStage = Literal[
    "baseline_contradiction",
    "baseline_unrecorded",
    "baseline_confirmed",
    "minicheck",
    "minicheck_unavailable",
]


@dataclass(frozen=True)
class SentenceVerdict:
    sentence: str
    stage: SentenceStage
    grounded: bool
    detail: str
    minicheck_score: float | None = None


@dataclass(frozen=True)
class CascadeVerdict:
    sentence_verdicts: tuple[SentenceVerdict, ...]

    @property
    def grounded(self) -> bool:
        """Empty is deliberately `False` - same fail-closed default as
        `grounding.GroundingVerdict.grounded`: a letter with no extractable
        sentences is not evidence of a clean letter."""
        if not self.sentence_verdicts:
            return False
        return all(v.grounded for v in self.sentence_verdicts)

    @property
    def ungrounded_verdicts(self) -> tuple[SentenceVerdict, ...]:
        return tuple(v for v in self.sentence_verdicts if not v.grounded)

    @property
    def minicheck_failure(self) -> SentenceVerdict | None:
        """The single most useful MiniCheck verdict for the audit row: the
        lowest-scoring sentence that actually reached and was withheld by
        stage 2. `None` when nothing was withheld by MiniCheck specifically
        (nothing reached stage 2, everything that did passed, or a
        stage-1 finding withheld first and already carries its own detail
        in `failure_reason`).
        """
        candidates = [v for v in self.ungrounded_verdicts if v.stage == "minicheck"]
        if not candidates:
            return None
        return min(candidates, key=lambda v: v.minicheck_score)


@dataclass(frozen=True)
class CascadeGateResult:
    """Mirrors `grounding.GateResult`'s shape (letter/verdict/failure_reason)
    but is its own type, not a shared one - this cascade and the Groq gate
    are two independent, separately callable paths (see module docstring),
    and typing them together would couple what is deliberately not coupled.
    """

    letter: DraftedLetter
    verdict: CascadeVerdict | None
    failure_reason: str | None

    @property
    def withheld(self) -> bool:
        return self.letter.provenance is LetterProvenance.FAILED_GROUNDING


def _document_text(context: DisputeContext) -> str:
    """MiniCheck's `document` input (see `minicheck.py`'s module docstring
    for the full contract) - a plain field=value rendering of every
    `DisputeContext` field, the same shape used in the Phase 1 smoke test.
    """
    return (
        f"reason_code={context.reason_code}; amount={context.amount:.2f}; "
        f"avs_match={context.avs_match}; cvv_match={context.cvv_match}; "
        f"device_fingerprint_known={context.device_fingerprint_known}; "
        f"delivery_confirmed={context.delivery_confirmed}; "
        f"prior_order_count={context.prior_order_count}"
    )


def _classify_sentence(
    sentence: str, context: DisputeContext, minicheck: MiniCheckScorer
) -> SentenceVerdict:
    contradictions = field_contradictions(sentence, context) + numeric_contradictions(
        sentence, context
    )
    if contradictions:
        return SentenceVerdict(sentence, "baseline_contradiction", False, contradictions[0].detail)

    unrecorded = unrecorded_entities(sentence)
    if unrecorded:
        return SentenceVerdict(sentence, "baseline_unrecorded", False, unrecorded[0].detail)

    confirmations = field_confirmations(sentence, context) + numeric_confirmations(
        sentence, context
    )
    if confirmations:
        return SentenceVerdict(sentence, "baseline_confirmed", True, confirmations[0].detail)

    try:
        score = minicheck.score(_document_text(context), sentence)
    except MiniCheckUnavailableError as error:
        return SentenceVerdict(
            sentence, "minicheck_unavailable", False, f"minicheck unavailable: {error}"
        )

    grounded = score >= MINICHECK_THRESHOLD
    detail = f"minicheck score={score:.3f} (threshold {MINICHECK_THRESHOLD})"
    return SentenceVerdict(sentence, "minicheck", grounded, detail, minicheck_score=score)


def grade_letter_with_cascade(
    letter_text: str, context: DisputeContext, minicheck: MiniCheckScorer
) -> CascadeVerdict:
    """The cascade on its own, separated from the gate so a caller (or a
    future eval harness, mirroring `grounding.grade_letter`) can score a
    letter without provenance bookkeeping."""
    verdicts = tuple(
        _classify_sentence(sentence, context, minicheck) for sentence in sentences(letter_text)
    )
    return CascadeVerdict(verdicts)


def _withhold(letter: DraftedLetter) -> DraftedLetter:
    return DraftedLetter(
        letter_text=letter.letter_text,
        cites_evidence_types=letter.cites_evidence_types,
        provenance=LetterProvenance.FAILED_GROUNDING,
    )


def apply_grounding_cascade(
    letter: DraftedLetter, context: DisputeContext, minicheck: MiniCheckScorer
) -> CascadeGateResult:
    """Run the cascade over a drafted letter. Same one-directional contract
    as `grounding.apply_grounding_gate`: a letter that is not already
    submittable is returned untouched, and this can only ever move a letter
    from `MODEL` to `FAILED_GROUNDING`, never the reverse.
    """
    if not letter.submittable:
        return CascadeGateResult(letter=letter, verdict=None, failure_reason=None)

    verdict = grade_letter_with_cascade(letter.letter_text, context, minicheck)
    if verdict.grounded:
        return CascadeGateResult(letter=letter, verdict=verdict, failure_reason=None)

    ungrounded = verdict.ungrounded_verdicts
    reason = "; ".join(f"{v.stage}: {v.sentence[:60]!r} ({v.detail})" for v in ungrounded[:3])
    return CascadeGateResult(
        letter=_withhold(letter),
        verdict=verdict,
        failure_reason=reason or "no sentences could be extracted",
    )
