"""The drafted `explanation_letter` as an object that carries where it came
from, and the one gate that decides whether it may be submitted.

Why this module exists (2026-09-02 remediation, defects 0.1 and 0.2). Before
it, a letter was a bare `ExplanationLetterOutput` plus a `human_review_required`
boolean carried alongside it. Two things went wrong with that shape:

1. The boolean travelled *next to* the letter rather than *in* it. By the time
   the letter reached `disputedesk/client/razorpay.py`, it was a plain `str`
   and the flag was gone, so a deterministic fallback letter - whose own body
   says "it has not been reviewed by a person yet" - was filed with the card
   network under `action="submit"`.
2. The letter's schema allowed 4,000 characters while Razorpay's `summary`
   field accepts 1,000, and the client closed that gap with `summary[:1000]`.
   Silent truncation destroys exactly the evidence the packet exists to carry.

Both are fixed by the same mechanism: `provenance` is a required field set at
construction, `letter_text` is bounded by the network's real limit at
construction, and `require_submittable` is the only door to `action="submit"`.
It fails closed - anything that is not `provenance == "model"` raises, before
any network call.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

# Razorpay's contest endpoint documents `summary` as "It can have a maximum
# length of 1000 characters" - https://razorpay.com/docs/api/disputes/contest/,
# re-verified 2026-09-02. This is the *only* place that number is written down:
# the drafting prompt, the schema the model's output is validated against, and
# the request body all read it from here, so they cannot drift apart again.
NETWORK_SUMMARY_MAX_CHARS = 1000

# A letter shorter than this is not a letter. Unchanged from the original
# `ExplanationLetterOutput` bound.
LETTER_MIN_CHARS = 50


class LetterProvenance(Enum):
    """Where a letter's text came from. Set at construction, never inferred
    from the letter's content or re-derived downstream.

    Plain `Enum`, not `(str, Enum)`, for the same reason
    `disputedesk.policy.engine.Decision` is - see that class's note.
    """

    MODEL = "model"
    """Drafted by the LLM and validated against the output schema on the
    normal generation path. The only value that may be submitted."""

    FALLBACK = "fallback"
    """The deterministic template, produced because the LLM failed twice
    (SPEC.md §7 failure path 2). Human review, never submission."""

    LOW_CONFIDENCE = "low_confidence"
    """Text that had to be altered to fit the network limit, or is otherwise
    not the model's own validated output. Human review, never submission."""


class LetterNotSubmittableError(RuntimeError):
    """Raised when a letter that is not `provenance == "model"` reaches a
    submission path. Deliberately an error and not a silently-degraded
    request: the caller must route to human review explicitly.
    """


class DraftedLetter(BaseModel):
    """One `explanation_letter` evidence object, ready to be filed or not.

    Frozen: provenance cannot be raised to `MODEL` after the fact by a later
    code path. Length is validated here, at construction, so an over-limit
    letter never exists as a value in the first place - there is nothing left
    downstream to truncate.
    """

    model_config = ConfigDict(frozen=True)

    letter_text: str = Field(min_length=LETTER_MIN_CHARS, max_length=NETWORK_SUMMARY_MAX_CHARS)
    cites_evidence_types: tuple[str, ...]
    provenance: LetterProvenance

    @property
    def submittable(self) -> bool:
        return self.provenance is LetterProvenance.MODEL


def require_submittable(letter: DraftedLetter) -> DraftedLetter:
    """Return `letter` if it may be submitted to the card network; raise
    `LetterNotSubmittableError` otherwise.

    Every path that sets `action="submit"` goes through this function. It is
    called inside the API client itself, not only in the pipeline, so a future
    caller that reaches the client directly still cannot file an unreviewed
    letter.
    """
    if not letter.submittable:
        raise LetterNotSubmittableError(
            f"letter provenance is {letter.provenance.value!r}, not 'model'; "
            "it must be routed to human review, not submitted"
        )
    return letter


def fallback_text_and_provenance(text: str) -> tuple[str, LetterProvenance]:
    """Shorten `text` to the network limit if it exceeds it, reporting the
    provenance that shortening implies.

    The one place in this codebase allowed to shorten letter text, and it
    never returns `MODEL`: a shortened letter is by definition no longer the
    validated output it started as, so it inherits the submission ban above.
    Used only by the deterministic fallback, whose template text is well
    inside the limit for every input this system produces - the branch exists
    so that an unusually long reason code or amount degrades honestly instead
    of raising a `ValidationError` in a failure path.
    """
    if len(text) <= NETWORK_SUMMARY_MAX_CHARS:
        return text, LetterProvenance.FALLBACK
    return text[: NETWORK_SUMMARY_MAX_CHARS - 3] + "...", LetterProvenance.LOW_CONFIDENCE
