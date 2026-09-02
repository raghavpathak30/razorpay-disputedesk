"""The grounding gate: a drafted letter may not be filed until every factual
assertion in it has been traced back to a field of the dispute record.

Why this exists. Until this module, a letter that validated against
`ExplanationLetterOutput` was submittable. That schema checks shape and
length - it cannot check whether the letter is *true*. A model that asserts
"the parcel was signed for by R. Sharma" produces a perfectly valid letter
about a signature this system has no field for, and it went to the card
network.

Two failure classes, and only the second justifies a model here:

- **Contradiction** - the letter asserts something a record field denies.
  Enumerable, and a deterministic field-matcher handles it
  (`eval/grounding_baseline.py` is that matcher, and is the baseline this
  gate is measured against).
- **Unrecorded assertion** - the letter asserts a fact the record has no
  field for *at all*. A deterministic checker validates the fields it
  enumerates; it cannot enumerate what the model invented, because the set of
  inventable facts is neither finite nor known in advance. This is the case
  the gate exists for, and the case the README's claim rests on.

**The gate is one-directional.** It can move a letter from `MODEL` to
`FAILED_GROUNDING`. It can never move a letter toward submission: a letter
that arrives non-`MODEL` is returned untouched without an LLM call, and no
path in this module constructs a letter with `LetterProvenance.MODEL`. It
never sees `p_win`, never sees the policy branch, and produces nothing
`policy/` has an input slot for - the isolation in CLAUDE.md invariant 4 is
preserved by construction, not by discipline.

**It fails closed.** A grader that raises, times out, returns malformed JSON
twice, violates the schema, or invents a field name all land on
`FAILED_GROUNDING`. The only way to stay submittable is for the grader to
affirmatively return a verdict in which every assertion is supported.
"""

import logging
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from disputedesk.evidence.context import DisputeContext
from disputedesk.evidence.letter import DraftedLetter, LetterProvenance
from disputedesk.evidence.llm import LLMClient
from disputedesk.evidence.prompts import load_prompt
from disputedesk.evidence.validated_call import call_llm_and_validate

logger = logging.getLogger(__name__)

PROMPT_VERSION = "grounding_gate_v1"

RECORD_FIELDS: frozenset[str] = frozenset(
    {
        "reason_code",
        "amount",
        "avs_match",
        "cvv_match",
        "device_fingerprint_known",
        "delivery_confirmed",
        "prior_order_count",
    }
)
"""Every field of `DisputeContext`, and the only values the grader may name in
`supporting_field`. Frozen here rather than derived from `DisputeContext`'s
annotations so that adding a field to the context does not silently widen what
the grader is allowed to claim support from - `tests/test_evidence_grounding.py`
asserts the two agree, so they can only diverge loudly."""

MAX_ASSERTIONS = 40
"""A bound on how much a grader can return. A letter is capped at 1,000
characters (`NETWORK_SUMMARY_MAX_CHARS`); forty factual claims in that space is
already far past anything a real letter contains, so a response longer than
this is a runaway generation, not a thorough audit."""

_QUOTE_MAX_CHARS = 200


class AssertionVerdict(BaseModel):
    """One factual claim extracted from the letter, and whether the record
    supports it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    quote: str = Field(min_length=1, max_length=_QUOTE_MAX_CHARS)
    supporting_field: str | None
    verdict: Literal["supported", "contradicted", "unsupported"]

    @model_validator(mode="after")
    def _field_must_match_verdict(self) -> "AssertionVerdict":
        """The three verdicts imply different things about `supporting_field`,
        and a response that gets that pairing wrong has not understood the
        task - so it is a schema violation, which fails closed, rather than a
        value to interpret charitably.

        `supported` and `contradicted` both say the record has a field about
        this claim, so both must name it. `unsupported` says the record has no
        such field, so naming one is a contradiction in terms.
        """
        if self.supporting_field is not None and self.supporting_field not in RECORD_FIELDS:
            raise ValueError(
                f"supporting_field {self.supporting_field!r} is not a record field; "
                f"expected one of {sorted(RECORD_FIELDS)} or null"
            )
        if self.verdict == "unsupported" and self.supporting_field is not None:
            raise ValueError("an 'unsupported' assertion must not name a supporting_field")
        if self.verdict != "unsupported" and self.supporting_field is None:
            raise ValueError(
                f"a {self.verdict!r} assertion must name the record field it refers to"
            )
        return self


class GroundingVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    assertions: list[AssertionVerdict] = Field(max_length=MAX_ASSERTIONS)

    @property
    def grounded(self) -> bool:
        """True only if the grader found at least one factual claim and every
        one of them is supported.

        The empty case is deliberately False. An empty verdict means the
        grader found nothing to check, which is not evidence that the letter
        is clean - treating it as a pass would make "return no assertions" the
        cheapest way for a grader (or for text inside the letter instructing
        one) to wave anything through.
        """
        if not self.assertions:
            return False
        return all(a.verdict == "supported" for a in self.assertions)

    @property
    def ungrounded_assertions(self) -> tuple[AssertionVerdict, ...]:
        return tuple(a for a in self.assertions if a.verdict != "supported")


@dataclass(frozen=True)
class GateResult:
    """`letter` is what the caller must use from here on - either the letter it
    passed in, unchanged, or the same text carrying `FAILED_GROUNDING`.

    `verdict` is `None` exactly when the grader could not produce one, in
    which case `failure_reason` says why. Both are carried for the audit row:
    "the gate withheld this" and "the gate could not run" are different facts
    about a dispute and a person reviewing the queue needs to tell them apart.
    """

    letter: DraftedLetter
    verdict: GroundingVerdict | None
    failure_reason: str | None

    @property
    def withheld(self) -> bool:
        return self.letter.provenance is LetterProvenance.FAILED_GROUNDING


def _withhold(letter: DraftedLetter) -> DraftedLetter:
    return DraftedLetter(
        letter_text=letter.letter_text,
        cites_evidence_types=letter.cites_evidence_types,
        provenance=LetterProvenance.FAILED_GROUNDING,
    )


def build_prompt(letter: DraftedLetter, context: DisputeContext) -> str:
    return load_prompt(PROMPT_VERSION).format(
        reason_code=context.reason_code,
        amount=f"{context.amount:.2f}",
        avs_match=context.avs_match,
        cvv_match=context.cvv_match,
        device_fingerprint_known=context.device_fingerprint_known,
        delivery_confirmed=context.delivery_confirmed,
        prior_order_count=context.prior_order_count,
        letter_text=letter.letter_text,
    )


def grade_letter(
    letter: DraftedLetter, context: DisputeContext, llm_client: LLMClient
) -> tuple[GroundingVerdict | None, str | None]:
    """The grader call on its own, separated from the gate so the eval harness
    can score verdicts without going through provenance bookkeeping.

    Returns `(verdict, None)` on success and `(None, reason)` on any failure.
    Catches broadly and deliberately: a grader that cannot be reached must
    withhold the letter, not raise into the caller and take the dispute's
    whole pipeline run down with it (SPEC.md §7 - the system degrades, it does
    not crash).
    """
    prompt = build_prompt(letter, context)
    try:
        parsed = call_llm_and_validate(llm_client, prompt, GroundingVerdict)
    except Exception as error:  # noqa: BLE001 - any grader failure must fail closed
        logger.info("grounding gate could not reach a verdict: %s", error)
        return None, f"grader call failed: {error}"
    if parsed is None:
        return None, "grader output failed schema validation twice"
    return parsed, None


def apply_grounding_gate(
    letter: DraftedLetter, context: DisputeContext, llm_client: LLMClient
) -> GateResult:
    """Run the gate over a drafted letter.

    A letter that is not already submittable is returned untouched and costs
    no LLM call - there is nothing to withhold, and raising it to submittable
    is the one thing this gate must never do.
    """
    if not letter.submittable:
        return GateResult(letter=letter, verdict=None, failure_reason=None)

    verdict, failure_reason = grade_letter(letter, context, llm_client)
    if verdict is None:
        return GateResult(
            letter=_withhold(letter),
            verdict=None,
            failure_reason=failure_reason,
        )
    if verdict.grounded:
        return GateResult(letter=letter, verdict=verdict, failure_reason=None)

    ungrounded = verdict.ungrounded_assertions
    reason = "; ".join(f"{a.verdict}: {a.quote[:60]!r}" for a in ungrounded[:3])
    return GateResult(
        letter=_withhold(letter),
        verdict=verdict,
        failure_reason=reason or "no factual assertions could be extracted",
    )
