"""The append-only audit log's only write and read paths. Every function here
either inserts a new row or reads existing ones - no `UPDATE` or `DELETE`
statement exists in this module.

That was, until 2026-09-02, the whole of the append-only guarantee. It is now
the *least* of it: the database refuses UPDATE and DELETE on these tables
outright (`disputedesk/audit/db.py`), and every row commits to its predecessor
(`disputedesk/audit/chain.py`). This module's job in that scheme is to compute
each new row's place in the chain, and to handle the one race the chain
introduces.
"""

import json
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from disputedesk.audit.chain import compute_row_hash, latest_hash
from disputedesk.audit.models import ApiOutcome, ChainedRecord, DecisionRecord, now_utc

_MAX_CHAIN_ATTEMPTS = 5
"""How many times an insert re-reads the chain tail after losing a race.

A fork is impossible because `prev_hash` is UNIQUE, so two writers that read
the same tail produce one commit and one `IntegrityError`; the loser re-reads
and appends after the winner. The bound exists so a pathological write storm
surfaces as an error rather than an unbounded loop - it is not a correctness
parameter, and hitting it means contention far beyond anything this system's
one-row-per-dispute write pattern produces.
"""


class ChainContentionError(RuntimeError):
    """Raised when an audit insert lost `_MAX_CHAIN_ATTEMPTS` races for the
    chain tail. The decision is *not* recorded - the caller must not proceed
    to file, since "persisted before the API call" is the invariant that makes
    the whole pipeline safe to retry.
    """


def get_decision(session: Session, dispute_id: str) -> DecisionRecord | None:
    return session.execute(
        select(DecisionRecord).where(DecisionRecord.dispute_id == dispute_id)
    ).scalar_one_or_none()


def _link_into_chain(session: Session, row: ChainedRecord) -> None:
    """Set `row`'s `prev_hash` from the current chain tail and compute its
    `row_hash`. Called immediately before each insert attempt, so a retry
    after a lost race re-reads the tail rather than re-using a stale one.
    """
    # `created_at` is part of the hashed payload, so it must be a real value
    # before the digest is taken - the column's `default=` would not fire
    # until flush, hashing a `None` the verifier would never recompute.
    if row.created_at is None:
        row.created_at = now_utc()
    row.prev_hash = latest_hash(session, type(row))
    row.row_hash = compute_row_hash(row.prev_hash, row.chain_payload())


def record_decision(
    session: Session,
    *,
    dispute_id: str,
    reason_code: str,
    amount_inr: float,
    model_version: str,
    features: dict,
    p_win: float,
    policy_branch: str,
    expected_value_inr: float,
    representment_cost_inr: float,
    low_confidence: bool,
    prompt_version: str | None,
    validation_result: str,
    human_review_required: bool,
) -> tuple[DecisionRecord, bool]:
    """Insert one decision row before the Razorpay API is ever touched
    (PHASES.md Phase 4 item 3). Returns `(row, was_newly_created)`.

    The check below is only a fast path; the real idempotency guarantee is
    the `dispute_id` UNIQUE constraint caught by `except IntegrityError`, so
    a request racing past the check still only lets one `INSERT` succeed.
    """
    existing = get_decision(session, dispute_id)
    if existing is not None:
        return existing, False

    row = DecisionRecord(
        dispute_id=dispute_id,
        reason_code=reason_code,
        amount_inr=amount_inr,
        model_version=model_version,
        features_json=json.dumps(features, sort_keys=True),
        p_win=p_win,
        policy_branch=policy_branch,
        expected_value_inr=expected_value_inr,
        representment_cost_inr=representment_cost_inr,
        low_confidence=low_confidence,
        prompt_version=prompt_version,
        validation_result=validation_result,
        human_review_required=human_review_required,
    )
    return _insert_chained(session, row, lambda: get_decision(session, dispute_id))


def _insert_chained(session, row, find_existing):
    """Insert `row`, retrying only the chain-tail race.

    Two different `IntegrityError`s can land here and they mean opposite
    things. A `dispute_id` collision means another request already recorded
    this dispute - that is the idempotency guarantee working, and the existing
    row is returned with `was_newly_created=False`. A `prev_hash` collision
    means another request appended to the chain first; nothing about *this*
    dispute has been recorded, so the row is re-linked to the new tail and
    retried. `find_existing()` tells the two apart without parsing the
    driver's error text.
    """
    for _ in range(_MAX_CHAIN_ATTEMPTS):
        _link_into_chain(session, row)
        session.add(row)
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            existing = find_existing()
            if existing is not None:
                return existing, False
            continue
        return row, True

    raise ChainContentionError(
        f"could not append to the audit chain after {_MAX_CHAIN_ATTEMPTS} attempts; "
        "nothing was recorded and nothing must be filed"
    )


def get_api_outcome(session: Session, dispute_id: str) -> ApiOutcome | None:
    return session.execute(
        select(ApiOutcome).where(ApiOutcome.dispute_id == dispute_id)
    ).scalar_one_or_none()


def record_api_outcome(
    session: Session,
    *,
    dispute_id: str,
    action: str,
    outcome: str,
    response: dict | None = None,
    error: str | None = None,
) -> ApiOutcome:
    """Insert the (at most one) filing outcome for `dispute_id`. Same
    idempotency mechanics as `record_decision`: a `dispute_id` UNIQUE
    constraint on `api_outcomes` means a second call for an already-filed
    dispute cannot create a second row - it returns the existing one,
    proving a retried/replayed call never double-files.
    """
    existing = get_api_outcome(session, dispute_id)
    if existing is not None:
        return existing

    row = ApiOutcome(
        dispute_id=dispute_id,
        action=action,
        outcome=outcome,
        response_json=json.dumps(response, sort_keys=True) if response is not None else None,
        error=error,
    )
    inserted, _was_new = _insert_chained(session, row, lambda: get_api_outcome(session, dispute_id))
    return inserted


@dataclass(frozen=True)
class AuditTrail:
    decision: DecisionRecord
    api_outcome: ApiOutcome | None


def get_audit_trail(session: Session, dispute_id: str) -> AuditTrail | None:
    """The decision row and its (possibly not-yet-existing) filing outcome,
    joined by `dispute_id` for display - e.g. the demo script's printed
    trail - without ever combining them into one mutated row.
    """
    decision = get_decision(session, dispute_id)
    if decision is None:
        return None
    return AuditTrail(decision=decision, api_outcome=get_api_outcome(session, dispute_id))
