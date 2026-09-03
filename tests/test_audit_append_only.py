"""The audit log is append-only in the database, not only by convention
(remediation defect 0.4).

`disputedesk/audit/models.py` claimed both tables were "append-only by
construction ... no update or delete path exists anywhere in the codebase".
That was true of the codebase and false of the database: an ordinary session
could rewrite or delete a decision row, and the verification pass did exactly
that. "No caller does X" is a statement about today's callers; append-only is
a property of the store.

Two independent mechanisms, tested separately here:

1. `BEFORE UPDATE` / `BEFORE DELETE` triggers that abort the statement. Same
   mechanism class as the `dispute_id` UNIQUE constraint that already carries
   the idempotency guarantee.
2. A hash chain, so that a tamper performed *around* the triggers - by
   something with the rights to drop them - is still detectable after the
   fact. Triggers stop the ordinary path; the chain catches the privileged
   one.
"""

import json

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DatabaseError

from disputedesk.audit.chain import verify_chain
from disputedesk.audit.db import get_engine, init_db, make_session_factory
from disputedesk.audit.log import record_api_outcome, record_decision
from disputedesk.audit.models import ApiOutcome, DecisionRecord


@pytest.fixture
def engine():
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    return engine


@pytest.fixture
def session(engine):
    s = make_session_factory(engine)()
    yield s
    s.close()


def _decision_kwargs(dispute_id: str, **overrides) -> dict:
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
        prompt_version="normalize_comms_log_v1,explanation_letter_v2",
        validation_result="validated",
        human_review_required=False,
    )
    kwargs.update(overrides)
    return kwargs


def _write_a_few_rows(session) -> None:
    for i in range(3):
        record_decision(session, **_decision_kwargs(f"disp_{i}"))
        record_api_outcome(
            session,
            dispute_id=f"disp_{i}",
            action="contest",
            outcome="success",
            response={"id": f"disp_{i}", "status": "under_review"},
        )


def _drop_append_only_triggers(engine) -> None:
    """Stand-in for a connection with rights the application role should not
    have. The triggers are the application-level lock; this test is about
    what remains true once someone has the key to it.
    """
    with engine.begin() as connection:
        for table in ("decisions", "api_outcomes"):
            for verb in ("update", "delete"):
                connection.execute(text(f"DROP TRIGGER IF EXISTS {table}_no_{verb}"))


# --------------------------------------------------------------------------
# 1. The ordinary path: the database refuses
# --------------------------------------------------------------------------


def test_an_ordinary_session_cannot_update_a_decision_row(session):
    record_decision(session, **_decision_kwargs("disp_1"))

    row = session.query(DecisionRecord).one()
    row.p_win = 0.99
    with pytest.raises(DatabaseError):
        session.commit()
    session.rollback()

    assert session.query(DecisionRecord).one().p_win == 0.8


def test_an_ordinary_session_cannot_delete_a_decision_row(session):
    record_decision(session, **_decision_kwargs("disp_1"))

    session.delete(session.query(DecisionRecord).one())
    with pytest.raises(DatabaseError):
        session.commit()
    session.rollback()

    assert session.query(DecisionRecord).count() == 1


def test_an_ordinary_session_cannot_update_an_api_outcome_row(session):
    record_decision(session, **_decision_kwargs("disp_1"))
    record_api_outcome(session, dispute_id="disp_1", action="contest", outcome="failed")

    row = session.query(ApiOutcome).one()
    row.outcome = "success"
    with pytest.raises(DatabaseError):
        session.commit()
    session.rollback()

    assert session.query(ApiOutcome).one().outcome == "failed"


def test_an_ordinary_session_cannot_delete_an_api_outcome_row(session):
    record_decision(session, **_decision_kwargs("disp_1"))
    record_api_outcome(session, dispute_id="disp_1", action="contest", outcome="failed")

    session.delete(session.query(ApiOutcome).one())
    with pytest.raises(DatabaseError):
        session.commit()
    session.rollback()

    assert session.query(ApiOutcome).count() == 1


def test_a_raw_sql_update_is_refused_too(session):
    """Not only the ORM path - the trigger is on the table, so a hand-written
    statement hits it as well.
    """
    record_decision(session, **_decision_kwargs("disp_1"))

    with pytest.raises(DatabaseError):
        session.execute(text("UPDATE decisions SET p_win = 0.99"))
        session.commit()
    session.rollback()


# --------------------------------------------------------------------------
# 2. The chain: a tamper around the triggers is still detectable
# --------------------------------------------------------------------------


def test_each_row_stores_its_predecessors_hash(session):
    _write_a_few_rows(session)

    rows = session.query(DecisionRecord).order_by(DecisionRecord.id).all()
    assert rows[0].prev_hash is None  # genesis
    assert rows[1].prev_hash == rows[0].row_hash
    assert rows[2].prev_hash == rows[1].row_hash
    assert len({r.row_hash for r in rows}) == 3


def test_verify_chain_passes_on_an_untampered_log(session):
    _write_a_few_rows(session)

    result = verify_chain(session)

    assert result.ok is True
    assert result.problems == []
    assert result.rows_checked == 6  # 3 decisions + 3 api outcomes


def test_verify_chain_passes_on_an_empty_log(session):
    result = verify_chain(session)

    assert result.ok is True
    assert result.rows_checked == 0


def test_verify_chain_fails_when_a_row_is_mutated_privileged(session, engine):
    _write_a_few_rows(session)
    _drop_append_only_triggers(engine)

    with engine.begin() as connection:
        connection.execute(text("UPDATE decisions SET p_win = 0.99 WHERE dispute_id = 'disp_1'"))
    session.expire_all()

    result = verify_chain(session)

    assert result.ok is False
    assert any("disp_1" in problem for problem in result.problems)


def test_verify_chain_fails_when_a_row_is_deleted_privileged(session, engine):
    _write_a_few_rows(session)
    _drop_append_only_triggers(engine)

    with engine.begin() as connection:
        connection.execute(text("DELETE FROM decisions WHERE dispute_id = 'disp_1'"))
    session.expire_all()

    result = verify_chain(session)

    assert result.ok is False


def test_verify_chain_fails_when_a_row_hash_is_rewritten_to_match_the_tamper(session, engine):
    """The hardest case: the attacker updates the row *and* recomputes that
    row's own `row_hash` so it is self-consistent. The next row's `prev_hash`
    still points at the old value, so the break moves rather than disappears.
    """
    _write_a_few_rows(session)
    _drop_append_only_triggers(engine)

    rows = session.query(DecisionRecord).order_by(DecisionRecord.id).all()
    tampered = rows[0]
    tampered.p_win = 0.99
    from disputedesk.audit.chain import compute_row_hash

    forged = compute_row_hash(None, tampered.chain_payload())

    with engine.begin() as connection:
        connection.execute(
            text("UPDATE decisions SET p_win = 0.99, row_hash = :h WHERE id = :i"),
            {"h": forged, "i": rows[0].id},
        )
    session.rollback()
    session.expire_all()

    result = verify_chain(session)

    assert result.ok is False


def test_the_chain_covers_the_features_json_a_reviewer_would_check(session, engine):
    """The audit row exists so a person can reconstruct why a decision was
    made. Silently swapping the recorded feature vector must break the chain.
    """
    _write_a_few_rows(session)
    _drop_append_only_triggers(engine)

    with engine.begin() as connection:
        connection.execute(
            text("UPDATE decisions SET features_json = :f WHERE dispute_id = 'disp_2'"),
            {"f": json.dumps({"amount": 1.0})},
        )
    session.expire_all()

    assert verify_chain(session).ok is False
