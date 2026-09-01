"""The append-only audit log's only write and read paths (CLAUDE.md: "no
update or delete path exists"). Every function here either inserts a new row
or reads existing ones - nothing calls `UPDATE` or `DELETE` anywhere in this
module, by construction, not by convention alone.
"""

import json
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from disputedesk.audit.models import ApiOutcome, DecisionRecord


def get_decision(session: Session, dispute_id: str) -> DecisionRecord | None:
    return session.execute(
        select(DecisionRecord).where(DecisionRecord.dispute_id == dispute_id)
    ).scalar_one_or_none()


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
    """Insert one decision row - PHASES.md Phase 4 item 3: called before the
    Razorpay API is ever touched for this dispute. Returns `(row,
    was_newly_created)`.

    A fast-path existence check runs first purely to avoid doing the insert
    (and, for the caller, the LLM calls that produced these arguments) for a
    dispute already decided. The actual idempotency guarantee is the
    `dispute_id` UNIQUE constraint on `decisions` (`disputedesk/audit/models.py`)
    caught below: even if two requests for the same dispute race past the
    fast-path check simultaneously, only one `INSERT` can ever succeed -
    this is the database enforcement PHASES.md Phase 4 item 2 asks for, not
    just this check.
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
    session.add(row)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        return get_decision(session, dispute_id), False  # type: ignore[return-value]
    return row, True


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
    session.add(row)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        return get_api_outcome(session, dispute_id)  # type: ignore[return-value]
    return row


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
