"""Tests for the append-only audit log (PHASES.md Phase 4 items 2, 4):
idempotency enforced by a database uniqueness constraint, not just an
application-level check, and every write going through insert-only
functions.
"""

import json

import pytest
from sqlalchemy.exc import IntegrityError

from disputedesk.audit.db import get_engine, init_db, make_session_factory
from disputedesk.audit.log import (
    get_api_outcome,
    get_audit_trail,
    get_decision,
    record_api_outcome,
    record_decision,
)
from disputedesk.audit.models import DecisionRecord


@pytest.fixture
def session():
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    s = make_session_factory(engine)()
    yield s
    s.close()


def _decision_kwargs(dispute_id="disp_1", **overrides):
    kwargs = dict(
        dispute_id=dispute_id,
        reason_code="MC_4837",
        amount_inr=5000.0,
        model_version="lgbm-config-v1-seed42",
        features={"amount": 5000.0, "avs_match": True},
        p_win=0.8,
        policy_branch="contest",
        expected_value_inr=3600.0,
        representment_cost_inr=400.0,
        low_confidence=False,
        prompt_version="normalize_comms_log_v1,explanation_letter_v1",
        validation_result="validated",
        human_review_required=False,
    )
    kwargs.update(overrides)
    return kwargs


def test_record_decision_creates_a_new_row(session):
    row, was_new = record_decision(session, **_decision_kwargs())

    assert was_new is True
    assert row.dispute_id == "disp_1"
    assert row.p_win == 0.8
    assert json.loads(row.features_json) == {"amount": 5000.0, "avs_match": True}


def test_recording_the_same_dispute_id_twice_does_not_create_a_second_row(session):
    record_decision(session, **_decision_kwargs())
    row2, was_new2 = record_decision(session, **_decision_kwargs(p_win=0.99))

    assert was_new2 is False
    assert row2.p_win == 0.8  # the *original* decision, not the second call's arguments

    all_rows = session.query(DecisionRecord).all()
    assert len(all_rows) == 1


def test_dispute_id_uniqueness_is_enforced_by_the_database_not_just_the_helper(session):
    """Bypass `record_decision`'s application-level check entirely and
    insert a duplicate row directly - PHASES.md Phase 4 item 2 requires the
    database itself to make this impossible.
    """
    record_decision(session, **_decision_kwargs())

    duplicate = DecisionRecord(
        dispute_id="disp_1",
        reason_code="MC_4837",
        amount_inr=1.0,
        model_version="x",
        features_json="{}",
        p_win=0.1,
        policy_branch="accept",
        expected_value_inr=-1.0,
        representment_cost_inr=400.0,
        low_confidence=False,
        prompt_version=None,
        validation_result="not_applicable",
        human_review_required=False,
    )
    session.add(duplicate)
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_get_decision_returns_none_for_an_unknown_dispute(session):
    assert get_decision(session, "does_not_exist") is None


def test_record_api_outcome_creates_and_then_deduplicates(session):
    record_decision(session, **_decision_kwargs())

    outcome1 = record_api_outcome(
        session,
        dispute_id="disp_1",
        action="contest",
        outcome="success",
        response={"id": "disp_1", "status": "under_review"},
    )
    outcome2 = record_api_outcome(
        session,
        dispute_id="disp_1",
        action="contest",
        outcome="failed",  # ignored - a row already exists for this dispute
        error="should not be recorded",
    )

    assert outcome1.outcome == "success"
    assert outcome2.outcome == "success"
    assert outcome1.id == outcome2.id
    assert get_api_outcome(session, "disp_1") is not None


def test_get_audit_trail_joins_decision_and_outcome(session):
    record_decision(session, **_decision_kwargs())
    record_api_outcome(
        session,
        dispute_id="disp_1",
        action="contest",
        outcome="success",
        response={"id": "disp_1", "status": "under_review"},
    )

    trail = get_audit_trail(session, "disp_1")

    assert trail.decision.dispute_id == "disp_1"
    assert trail.api_outcome.outcome == "success"


def test_get_audit_trail_handles_no_outcome_yet(session):
    record_decision(session, **_decision_kwargs(policy_branch="escalate"))

    trail = get_audit_trail(session, "disp_1")

    assert trail.decision.policy_branch == "escalate"
    assert trail.api_outcome is None


def test_get_audit_trail_returns_none_for_an_unknown_dispute(session):
    assert get_audit_trail(session, "does_not_exist") is None
