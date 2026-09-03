"""Razorpay's published chargeback reason-code list, and the two questions the
system asks of an incoming code: is it real, and do we have a strategy for it.

Why this module exists (2026-09-02 remediation, defect 0.3). The webhook
constrained `reason_code` to a hand-typed four-value `Literal` built from
`disputedesk.features.build.REASON_CODES`. Anything else - including Visa 83,
a code on Razorpay's own published reference and the very code `GENERATOR.md`
§8 cites as this project's Visa source - was rejected with HTTP 422.

That conflated two different facts:

- **Malformed**: the payload is not a dispute. A 422 is right.
- **Unrecognised**: the payload is a perfectly good dispute carrying a code
  this system has no defense strategy for. A 422 is wrong - it makes the
  dispute disappear at the boundary with no audit row and no queue entry,
  which is the worst possible outcome for a real chargeback on a real
  deadline.

So the webhook no longer gates on reason code at all. Every code is accepted;
`is_supported_reason_code` then decides whether the dispute goes down the
normal path or is tagged `reason_code_unrecognised` and queued for a person
without anything being filed in either direction
(`disputedesk/api/pipeline.py`).

Three sets, deliberately kept distinct:

- `PUBLISHED_REASON_CODES` - what Razorpay publishes. Read from the committed
  fixture, not typed into this file, so "did we narrow the accepted set" is a
  question about a data file with a source URL on it.
- `REQUIRED_EVIDENCE_BY_REASON_CODE` (`reason_code_map.py`) - the codes this
  project has an evidence strategy for. Four CNP-fraud codes: SPEC.md's "one
  class of loss, exactly one".
- `REASON_CODES` (`features/build.py`) - the model's categorical vocabulary,
  i.e. the codes the *generator* produces. Not touched here: what the model
  was trained on is a different fact from what the webhook accepts, and
  `build_features` already maps an unseen category past the end of its
  vocabulary rather than raising.
"""

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from disputedesk.evidence.reason_code_map import (
    REQUIRED_EVIDENCE_BY_REASON_CODE,
    canonical_reason_code,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "reference" / "razorpay_chargeback_codes.csv"
)


@dataclass(frozen=True)
class PublishedReasonCode:
    network: str
    code: str
    wire_code: str
    reason_text: str
    category: str
    preventable: bool
    reversible: bool


def _load() -> tuple[PublishedReasonCode, ...]:
    with FIXTURE_PATH.open(newline="", encoding="utf-8") as handle:
        rows = [line for line in handle if not line.startswith("#")]
    return tuple(
        PublishedReasonCode(
            network=row["network"],
            code=row["code"],
            wire_code=row["wire_code"],
            reason_text=row["reason_text"],
            category=row["category"],
            preventable=row["preventable"] == "YES",
            reversible=row["reversible"] == "YES",
        )
        for row in csv.DictReader(rows)
    )


PUBLISHED_REASON_CODES: tuple[PublishedReasonCode, ...] = _load()

PUBLISHED_WIRE_CODES: frozenset[str] = frozenset(c.wire_code for c in PUBLISHED_REASON_CODES)


def is_supported_reason_code(reason_code: str) -> bool:
    """True if this system has an evidence strategy for `reason_code` - i.e.
    a dispute carrying it can be assembled and filed.

    False covers two different situations that are handled identically, and
    are identical in what they should cause: a published code outside this
    project's one loss class (e.g. `MC_4855`, Non-Receipt of Merchandise), and
    a code that is not a code at all. Both mean "this system should not act on
    this dispute unsupervised", which is the only decision this function is
    asked to inform. `PUBLISHED_WIRE_CODES` is available for a caller that
    needs to tell the two apart for reporting.
    """
    return canonical_reason_code(reason_code) in REQUIRED_EVIDENCE_BY_REASON_CODE


@lru_cache
def fraud_category_wire_codes() -> frozenset[str]:
    """Every published code Razorpay marks `Chargeback Category: Fraud`.

    Not used to decide anything - `is_supported_reason_code` is deliberately
    the narrower question, because being fraud-category is not the same as
    this project having an evidence strategy for it. Exposed because
    GENERATOR.md §8's scoping argument rests on this set, and a reader should
    be able to recompute it from the fixture rather than take the prose on
    trust.
    """
    return frozenset(c.wire_code for c in PUBLISHED_REASON_CODES if c.category == "Fraud")
