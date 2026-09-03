"""The audit log's hash chain: each row commits to its predecessor, so a
tamper is detectable after the fact even by someone who could get around the
append-only triggers.

Why this exists (2026-09-02 remediation, defect 0.4). The append-only claim
rested on "no update or delete path exists anywhere in the codebase". That is
a statement about today's callers, not about the store, and the verification
pass falsified it by rewriting and deleting a row through an ordinary session.
Two defences were added, and they answer different threats:

- `BEFORE UPDATE`/`BEFORE DELETE` triggers (`disputedesk/audit/db.py`) stop
  the ordinary path - a bug, a stray script, a future code path.
- This chain stops the ordinary path being the only thing that matters.
  Anything with the rights to drop a trigger can still edit the table; what it
  cannot do is edit it *invisibly*, because every later row's `prev_hash`
  commits to the edited row's content.

The chain is not tamper-*proof*. Someone who can rewrite every row from the
edit point forward, in order, can produce a self-consistent chain. What it
buys is that a tamper is no longer a single silent `UPDATE` - it is a rewrite
of the entire suffix of the log, which is a much larger and noisier act, and
which an off-box copy of any single `row_hash` would still catch. Stated
plainly rather than overclaimed.
"""

import hashlib
import json
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from disputedesk.audit.models import CHAINED_MODELS, ChainedRecord


def compute_row_hash(prev_hash: str | None, payload: dict) -> str:
    """The row's hash: SHA-256 over its predecessor's hash and its own
    business fields.

    `payload` is serialised with `sort_keys=True` and `default=str` so the
    digest depends on the row's *values*, not on dict ordering or on a
    datetime's repr changing between Python versions. `prev_hash` is folded in
    as an explicit field rather than concatenated, so a value containing the
    separator cannot be used to shift the boundary between the two parts.
    """
    material = json.dumps(
        {"prev_hash": prev_hash, "row": payload}, sort_keys=True, default=str
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def latest_hash(session: Session, model: type[ChainedRecord]) -> str | None:
    """The `row_hash` of the newest row in `model`'s chain, or `None` if the
    chain is empty (the next row written will be its genesis).
    """
    return session.execute(
        select(model.row_hash).order_by(model.id.desc()).limit(1)
    ).scalar_one_or_none()


@dataclass(frozen=True)
class ChainVerification:
    ok: bool
    rows_checked: int
    problems: list[str]


def _verify_one(session: Session, model: type[ChainedRecord]) -> tuple[int, list[str]]:
    rows = session.execute(select(model).order_by(model.id)).scalars().all()
    problems: list[str] = []
    expected_prev: str | None = None

    for position, row in enumerate(rows):
        label = f"{model.__tablename__}[{position}] dispute_id={row.dispute_id}"

        if row.prev_hash != expected_prev:
            problems.append(
                f"{label}: prev_hash does not match the preceding row's row_hash "
                f"(a row was inserted, removed, or reordered)"
            )
        recomputed = compute_row_hash(row.prev_hash, row.chain_payload())
        if recomputed != row.row_hash:
            problems.append(f"{label}: row_hash does not match the row's contents (row edited)")

        expected_prev = row.row_hash

    return len(rows), problems


def verify_chain(session: Session) -> ChainVerification:
    """Walk every chained audit table in insertion order and check both links:
    that each row's stored `row_hash` still matches its own contents, and that
    each row's `prev_hash` matches its predecessor's `row_hash`.

    Returns a result rather than raising, because "is this log intact" is a
    question an operator asks, and the answer is more useful as a list of
    which rows broke than as a traceback.
    """
    total = 0
    problems: list[str] = []
    for model in CHAINED_MODELS:
        checked, found = _verify_one(session, model)
        total += checked
        problems.extend(found)
    return ChainVerification(ok=not problems, rows_checked=total, problems=problems)
