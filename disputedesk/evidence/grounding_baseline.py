"""The deterministic baseline the grounding gate is measured against: a
field-matcher over the seven record fields, plus a generic detector for
assertions that look like unrecorded entities.

Moved here from `eval/grounding_baseline.py` (CLAUDE.md Day-2 grounding-gate
cascade) because it now has a second job beyond eval-time comparison: it is
cascade stage 1 in `disputedesk/evidence/grounding_cascade.py`, run before
MiniCheck (stage 2) on every sentence. `eval/grounding_eval.py` and
`eval/run_grounding_eval.py` still import `baseline_flags` from here for the
already-published n45/clean27 comparison - that function's behaviour is
unchanged (see the 2026-09-08 DECISIONS.md entry on why the new
`*_confirmations` functions below are deliberately excluded from
`baseline_findings`/`baseline_flags`, so those recorded numbers do not move).

This is written to be the honest strong version of "just write a function",
not a strawman. If it beats the gate, that is the result and it goes in the
README. Two things follow from that intent:

- It gets **both** halves of the job. The field-matcher is the part a
  deterministic checker is naturally good at (Class A: the letter asserts
  something a record field denies). `_unrecorded_entity_hits` is a real
  attempt at the part it is naturally bad at (Class B: the letter asserts a
  fact no field covers), rather than leaving it with no mechanism at all.
- **Its scope limit is structural, not a tuning choice.** A field-matcher can
  only match fields. For Class B the baseline has to fall back to guessing
  which *shapes* of text tend to be unrecorded facts - tracking numbers,
  signatures, phone calls, dates - which is a fixed list against an open set.
  That asymmetry is the whole argument for the gate, and it is why this module
  is committed alongside it rather than described.

**Authorship disclosure.** This baseline, the corpus in
`eval/grounding_corpus.py`, and the gate's prompt were all authored by the
same person in the same session. A reader cannot verify that the injection
templates were not tuned to evade these patterns. What a reader *can* do is
read both files - the templates are committed verbatim, and their composition
is published in the README - and judge the difficulty for themselves. That is
the ceiling on any claim built from this comparison, and it cannot be raised
by running the comparison harder.

No LLM. No network.
"""

import re
from dataclasses import dataclass

from disputedesk.evidence.context import DisputeContext

_NEGATION = re.compile(
    r"\b(no|not|never|non|nor|without|failed|unable|could not|did not|was not|were not|"
    r"is not|are not|unconfirmed|unverified|unrecognis|unrecogniz|mismatch|discrepan)\w*\b",
    re.IGNORECASE,
)

# Topic patterns: "does this sentence talk about field X at all". Deliberately
# generous - a missed topic costs the baseline a detection, and this module is
# supposed to be the strong version.
_FIELD_TOPICS: dict[str, re.Pattern] = {
    "avs_match": re.compile(
        r"\b(address verification|avs|billing address (match|verif)|verified the address)\b",
        re.IGNORECASE,
    ),
    "cvv_match": re.compile(
        r"\b(card verification|cvv|cvc|security code|card code)\b", re.IGNORECASE
    ),
    "device_fingerprint_known": re.compile(
        r"\b(device|fingerprint|browser|handset|same machine|known device)\b", re.IGNORECASE
    ),
    "delivery_confirmed": re.compile(
        r"\b(deliver\w*|shipp\w*|dispatch\w*|parcel|consignment|fulfil\w*|fulfill\w*|"
        r"courier|received the (order|goods|item))\b",
        re.IGNORECASE,
    ),
}

_BOOLEAN_FIELDS: tuple[str, ...] = (
    "avs_match",
    "cvv_match",
    "device_fingerprint_known",
    "delivery_confirmed",
)

# Shapes of text that tend to be facts no record field covers. A fixed list
# against an open set - see the module docstring.
_UNRECORDED_SHAPES: dict[str, re.Pattern] = {
    "tracking_number": re.compile(r"\b(?=\w*\d)[A-Z0-9]{8,}\b|\btracking (number|id)\b", re.I),
    "signature": re.compile(r"\bsign(ed|ature)\b", re.IGNORECASE),
    "phone_contact": re.compile(
        r"\b(phone|telephone|called|call with|spoke (with|to)|by phone)\b", re.IGNORECASE
    ),
    "email_contact": re.compile(r"\b(e-?mail(ed)?|inbox|opened the message)\b", re.IGNORECASE),
    "explicit_date": re.compile(
        r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2} (January|February|March|April|May|June|July|"
        r"August|September|October|November|December)\b",
        re.IGNORECASE,
    ),
    "ip_or_login": re.compile(r"\b(ip address|logged in|login|session id)\b", re.IGNORECASE),
    "named_person": re.compile(r"\b(Mr|Mrs|Ms|Dr)\.? [A-Z][a-z]+\b|\b[A-Z]\. [A-Z][a-z]+\b"),
}


@dataclass(frozen=True)
class BaselineFinding:
    kind: str  # "contradiction" | "unrecorded" | "confirmed"
    detail: str


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def sentences(text: str) -> list[str]:
    """Public alias of `_sentences` for callers outside this module -
    `grounding_cascade.py` splits a letter into sentences with this exact
    function so stage 1 and stage 2 of the cascade agree on what "one
    sentence" means, rather than each maintaining its own splitter."""
    return _sentences(text)


def _asserted_polarity(sentence: str) -> bool:
    """True if the sentence asserts the positive form of whatever it is about.

    A single negation marker anywhere in the sentence flips it. This is the
    crude part, and it is crude on purpose: sentence-level negation scope is
    the thing a regex cannot get right, and pretending otherwise by
    hand-tuning around the corpus would be exactly the rigging this comparison
    exists to avoid.
    """
    return _NEGATION.search(sentence) is None


def field_contradictions(letter_text: str, context: DisputeContext) -> list[BaselineFinding]:
    """Sentences that assert the opposite polarity to a recorded boolean."""
    findings = []
    for field in _BOOLEAN_FIELDS:
        recorded = bool(getattr(context, field))
        topic = _FIELD_TOPICS[field]
        for sentence in _sentences(letter_text):
            if not topic.search(sentence):
                continue
            if _asserted_polarity(sentence) != recorded:
                findings.append(
                    BaselineFinding(
                        kind="contradiction",
                        detail=f"{field}={recorded} but letter says: {sentence[:70]!r}",
                    )
                )
                break
    return findings


def numeric_contradictions(letter_text: str, context: DisputeContext) -> list[BaselineFinding]:
    """The two numeric fields a letter is allowed to state. Exact-match only -
    a number the record does not hold is a contradiction."""
    findings = []
    amounts = {a.replace(",", "") for a in re.findall(r"INR\s*([\d,]+(?:\.\d{2})?)", letter_text)}
    for stated in amounts:
        try:
            if abs(float(stated) - context.amount) > 0.005:
                findings.append(
                    BaselineFinding(
                        "contradiction", f"amount={context.amount} but letter says {stated}"
                    )
                )
        except ValueError:
            continue

    for stated_count in re.findall(r"\b(\d+)\s+prior orders?\b", letter_text, re.IGNORECASE):
        if int(stated_count) != context.prior_order_count:
            findings.append(
                BaselineFinding(
                    "contradiction",
                    f"prior_order_count={context.prior_order_count} but letter says {stated_count}",
                )
            )
    return findings


def field_confirmations(letter_text: str, context: DisputeContext) -> list[BaselineFinding]:
    """Mirror of `field_contradictions`: same topic regex, same
    negation-polarity extraction, flipped comparison - a sentence that
    asserts the *same* polarity a recorded boolean already holds.

    Exists so the cascade (`grounding_cascade.py`) can resolve an exact
    boolean-field match at stage 1 instead of sending it to MiniCheck to
    guess at - see the 2026-09-08 DECISIONS.md entry: without this,
    `field_contradictions` returning `[]` was indistinguishable from "this
    sentence has nothing to do with any record field", and stage 1 had no
    way to ever pass a sentence, only fail one.
    """
    findings = []
    for field in _BOOLEAN_FIELDS:
        recorded = bool(getattr(context, field))
        topic = _FIELD_TOPICS[field]
        for sentence in _sentences(letter_text):
            if not topic.search(sentence):
                continue
            if _asserted_polarity(sentence) == recorded:
                findings.append(
                    BaselineFinding(
                        kind="confirmed",
                        detail=f"{field}={recorded}, letter agrees: {sentence[:70]!r}",
                    )
                )
                break
    return findings


def numeric_confirmations(letter_text: str, context: DisputeContext) -> list[BaselineFinding]:
    """Mirror of `numeric_contradictions`: the same two numeric fields,
    exact-match only, flipped to flag agreement rather than disagreement.
    See `field_confirmations`'s docstring for why this exists."""
    findings = []
    amounts = {a.replace(",", "") for a in re.findall(r"INR\s*([\d,]+(?:\.\d{2})?)", letter_text)}
    for stated in amounts:
        try:
            if abs(float(stated) - context.amount) <= 0.005:
                findings.append(
                    BaselineFinding(
                        "confirmed", f"amount={context.amount}, letter agrees: INR {stated}"
                    )
                )
        except ValueError:
            continue

    for stated_count in re.findall(r"\b(\d+)\s+prior orders?\b", letter_text, re.IGNORECASE):
        if int(stated_count) == context.prior_order_count:
            findings.append(
                BaselineFinding(
                    "confirmed",
                    f"prior_order_count={context.prior_order_count}, letter agrees: {stated_count}",
                )
            )
    return findings


def unrecorded_entities(letter_text: str) -> list[BaselineFinding]:
    """Text shaped like a fact the record has no field for."""
    return [
        BaselineFinding("unrecorded", f"{name}: {match.group(0)[:40]!r}")
        for name, pattern in _UNRECORDED_SHAPES.items()
        if (match := pattern.search(letter_text)) is not None
    ]


def baseline_findings(letter_text: str, context: DisputeContext) -> list[BaselineFinding]:
    """Deliberately excludes `field_confirmations`/`numeric_confirmations`.

    This function backs `baseline_flags`, which backs the already-published
    n45/clean27 eval numbers (NUMBERS.md, README) - CLAUDE.md's "do not
    silently change a number that is already recorded in a document". A real
    letter mentions delivery/AVS/amount routinely, so folding confirmations
    in here would flip `baseline_flags` to `True` on nearly every clean
    letter and inflate the baseline's measured false-flag rate for reasons
    that have nothing to do with the eval those numbers report. The
    confirmation functions are for `grounding_cascade.py` only.
    """
    return (
        field_contradictions(letter_text, context)
        + numeric_contradictions(letter_text, context)
        + unrecorded_entities(letter_text)
    )


def baseline_flags(letter_text: str, context: DisputeContext) -> bool:
    """The baseline's single binary decision, in the same shape as the gate's:
    True means "withhold this letter for review"."""
    return bool(baseline_findings(letter_text, context))
