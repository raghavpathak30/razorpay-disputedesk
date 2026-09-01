"""SQLAlchemy ORM models for the audit log (CLAUDE.md, PHASES.md Phase 4).

Two tables, both append-only by construction: `disputedesk/audit/log.py` -
the only module that writes to either - exposes insert-only functions and no
update or delete path exists anywhere in the codebase.

`DecisionRecord` is the audit row SPEC.md §1 step 6 asks for: one row per
dispute, written before the Razorpay API is ever called (PHASES.md Phase 4
item 3), carrying every field the phase's audit-row spec lists except the
API response. `dispute_id` is UNIQUE - the database-enforced idempotency
gate (PHASES.md: "Enforce this in the database, not in application logic"):
a second decision for the same dispute is a constraint violation, not just
an unlikely event.

`ApiOutcome` carries the eventual API response - written only after the
(possibly retried) API call finishes, success or exhausted-retries failure.
It is a separate table, not a later UPDATE of `DecisionRecord`, specifically
so `DecisionRecord` can be both "persisted before the API call" and never
touched again. `dispute_id` is UNIQUE here too, for the same reason: at most
one filing outcome can ever exist per dispute.
"""

from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _now() -> datetime:
    return datetime.now(UTC)


class DecisionRecord(Base):
    __tablename__ = "decisions"
    __table_args__ = (UniqueConstraint("dispute_id", name="uq_decisions_dispute_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dispute_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    reason_code: Mapped[str] = mapped_column(String, nullable=False)
    amount_inr: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    features_json: Mapped[str] = mapped_column(String, nullable=False)
    p_win: Mapped[float] = mapped_column(Float, nullable=False)
    policy_branch: Mapped[str] = mapped_column(String, nullable=False)
    expected_value_inr: Mapped[float] = mapped_column(Float, nullable=False)
    representment_cost_inr: Mapped[float] = mapped_column(Float, nullable=False)
    low_confidence: Mapped[bool] = mapped_column(nullable=False)
    prompt_version: Mapped[str | None] = mapped_column(String, nullable=True)
    validation_result: Mapped[str] = mapped_column(String, nullable=False)
    human_review_required: Mapped[bool] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class ApiOutcome(Base):
    __tablename__ = "api_outcomes"
    __table_args__ = (UniqueConstraint("dispute_id", name="uq_api_outcomes_dispute_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dispute_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    action: Mapped[str] = mapped_column(String, nullable=False)  # "accept" | "contest"
    outcome: Mapped[str] = mapped_column(String, nullable=False)  # "success" | "failed"
    response_json: Mapped[str | None] = mapped_column(String, nullable=True)
    error: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
