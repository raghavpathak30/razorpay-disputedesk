"""The frozen evaluation corpus for the grounding gate: clean letters, letters
that contradict the record, and letters that assert facts the record has no
field for.

Two positive classes, reported separately and never pooled, because they are
different tasks and pooling would let the easy one carry the hard one:

- **Class A - contradiction.** A recorded boolean is flipped *after* the
  letter was drafted, so the letter now asserts what the record denies. The
  ground truth is the flip: mechanical, with no authorial judgment in the
  label. A deterministic field-matcher should score well here and that is
  expected.
- **Class B - unrecorded assertion.** A sentence asserting a fact no record
  field covers is inserted into the letter. The ground truth is the insertion.
  These templates are authored, so their difficulty is a choice - see the
  disclosure below and in `eval/grounding_baseline.py`.

**The README's claim rests on Class B.** Class A is reported to show the gate
does not lose to the baseline on the easy half, not to support the claim.

**Class A field selection is deliberately not the baseline's own matcher.**
`_MENTIONS` below is an independent, simpler topic list. Selecting the field
to flip with `eval.grounding_baseline._FIELD_TOPICS` would make the baseline's
topic-match step true by construction on every Class A item, turning half its
score into an identity rather than a measurement.

**Authorship disclosure.** The insertion templates, the baseline's patterns
and the gate's prompt were authored by the same person in the same session.
The templates are committed verbatim below and their composition is published
in the README so a reader can judge the difficulty. Nothing here was edited
after seeing a score; the corpus is built from a fixed seed and pinned by
`tests/test_eval_grounding_corpus.py`.

No LLM. No network. The drafted letters this operates on come from a committed
recording (`data/reference/`), produced by `eval/run_grounding_draft.py`.
"""

import dataclasses
import re
from dataclasses import dataclass
from typing import Literal

import numpy as np

from disputedesk.evidence.context import DisputeContext

ItemClass = Literal["clean", "contradiction", "unrecorded"]

FLIPPABLE_FIELDS: tuple[str, ...] = (
    "avs_match",
    "cvv_match",
    "device_fingerprint_known",
    "delivery_confirmed",
)

_MENTIONS: dict[str, re.Pattern] = {
    "avs_match": re.compile(r"address|avs", re.IGNORECASE),
    "cvv_match": re.compile(r"cvv|cvc|verification value|security code", re.IGNORECASE),
    "device_fingerprint_known": re.compile(r"device|fingerprint|browser", re.IGNORECASE),
    "delivery_confirmed": re.compile(r"deliver|shipp|dispatch|parcel|fulfil", re.IGNORECASE),
}

# Sentences asserting a fact none of the seven record fields covers. Written as
# plausible model hallucinations for a chargeback letter - the kind of specific,
# confident detail a drafting model reaches for when the record is thin - not
# written to evade any particular detector. Some of these are shaped in ways a
# regex can catch and some are not; which is which is the measurement.
UNRECORDED_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("signature", "The parcel was signed for by R. Sharma at the delivery address."),
    (
        "tracking",
        "Tracking reference 1Z9X4A2210 shows the consignment reached the destination hub.",
    ),
    (
        "phone_call",
        "Our support team spoke with the cardholder by telephone and confirmed the order.",
    ),
    (
        "email_open",
        "A confirmation e-mail was sent to the registered address and was opened twice.",
    ),
    ("date", "The order was placed on 14 March and released from the warehouse the following day."),
    ("ip_login", "The account was logged in from the same IP address used for previous purchases."),
    ("loyalty", "The cardholder is enrolled in our loyalty programme and holds gold-tier status."),
    ("no_prior_disputes", "This customer has never previously raised a dispute against our store."),
    ("order_contents", "The order comprised two items, both of which were in stock and reserved."),
    ("refund_offer", "We contacted the customer to offer a partial refund, which was declined."),
    ("terms_accepted", "The customer accepted our published terms and conditions at checkout."),
    (
        "warehouse",
        "Our warehouse records show the item was picked, packed and scanned out for courier.",
    ),
)


@dataclass(frozen=True)
class CorpusItem:
    item_id: str
    letter_text: str
    context: DisputeContext
    item_class: ItemClass
    # For Class A, the field that was flipped. For Class B, the template id.
    # For clean items, None. This is the ground truth's provenance, carried so
    # a per-field or per-template breakdown can be reported.
    mutation: str | None

    @property
    def should_be_flagged(self) -> bool:
        return self.item_class != "clean"


def mentioned_fields(letter_text: str) -> tuple[str, ...]:
    """Which flippable record fields this letter actually talks about.

    Flipping a field the letter never mentions would produce an item labelled
    "contradiction" that contains no contradiction - a mislabelled positive
    that would punish both arms for not finding something that is not there.
    """
    return tuple(f for f in FLIPPABLE_FIELDS if _MENTIONS[f].search(letter_text))


def make_contradiction(
    letter_text: str, context: DisputeContext, rng: np.random.Generator
) -> tuple[DisputeContext, str] | None:
    """Flip one mentioned boolean in the record. Returns the mutated context
    and the field flipped, or None when the letter mentions no flippable
    field."""
    candidates = mentioned_fields(letter_text)
    if not candidates:
        return None
    field = str(rng.choice(np.array(candidates, dtype=object)))
    mutated = dataclasses.replace(context, **{field: not getattr(context, field)})
    return mutated, field


def _insertion_point(letter_text: str, rng: np.random.Generator) -> int:
    """Index of a sentence boundary to insert at. Inserting mid-letter rather
    than always appending stops position alone from being a giveaway."""
    boundaries = [m.end() for m in re.finditer(r"(?<=[.!?])\s+", letter_text)]
    if not boundaries:
        return len(letter_text)
    return int(rng.choice(np.array(boundaries)))


def make_unrecorded(letter_text: str, rng: np.random.Generator) -> tuple[str, str]:
    """Insert one unrecorded-fact sentence. Returns the new text and the
    template id."""
    index = int(rng.integers(0, len(UNRECORDED_TEMPLATES)))
    template_id, sentence = UNRECORDED_TEMPLATES[index]
    at = _insertion_point(letter_text, rng)
    return (letter_text[:at] + sentence + " " + letter_text[at:]).strip(), template_id


def build_corpus(drafts: list[tuple[str, DisputeContext]], seed: int = 0) -> list[CorpusItem]:
    """One clean item, one Class A item and one Class B item per draft.

    A draft that mentions no flippable field yields no Class A item; that is
    recorded by its absence rather than forced, so no item is labelled a
    contradiction it does not contain.
    """
    rng = np.random.default_rng(seed)
    items: list[CorpusItem] = []
    for i, (letter_text, context) in enumerate(drafts):
        items.append(CorpusItem(f"d{i:04d}_clean", letter_text, context, "clean", None))

        contradiction = make_contradiction(letter_text, context, rng)
        if contradiction is not None:
            mutated_context, field = contradiction
            items.append(
                CorpusItem(f"d{i:04d}_contra", letter_text, mutated_context, "contradiction", field)
            )

        mutated_text, template_id = make_unrecorded(letter_text, rng)
        items.append(
            CorpusItem(f"d{i:04d}_unrec", mutated_text, context, "unrecorded", template_id)
        )
    return items


def composition(items: list[CorpusItem]) -> dict:
    """The table the README publishes so a reader can judge the corpus's
    difficulty rather than take it on trust."""
    by_class: dict[str, int] = {}
    by_mutation: dict[str, int] = {}
    for item in items:
        by_class[item.item_class] = by_class.get(item.item_class, 0) + 1
        if item.mutation is not None:
            key = f"{item.item_class}:{item.mutation}"
            by_mutation[key] = by_mutation.get(key, 0) + 1
    return {"n_items": len(items), "by_class": by_class, "by_mutation": by_mutation}
