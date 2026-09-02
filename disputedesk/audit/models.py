"""SQLAlchemy ORM models for the audit log (CLAUDE.md, PHASES.md Phase 4).

Two tables, both append-only. Until 2026-09-02 that word rested on an
application-level claim - "`disputedesk/audit/log.py` exposes insert-only
functions and no update or delete path exists anywhere in the codebase" - which
the verification pass falsified by rewriting and deleting a row through an
ordinary session. It is now enforced by the database:

- `BEFORE UPDATE` and `BEFORE DELETE` triggers on both tables abort the
  statement (`disputedesk/audit/db.py`), the same mechanism class as the
  UNIQUE constraint that already carries idempotency.
- Every row commits to its predecessor via `prev_hash`/`row_hash`, so an edit
  made by something that *could* drop the triggers is still detectable
  afterwards (`disputedesk/audit/chain.py`).

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


def now_utc() -> datetime:
    """UTC, but tz-naive.

    The column is a plain `DateTime`, and SQLite's DATETIME storage format
    drops the offset - so a tz-aware value written here comes back naive on
    the next read. That round-trip asymmetry is invisible until a value is
    hashed: `chain_payload()` would commit to `...+00:00` at insert and
    recompute over `...` at verification, breaking the chain on rows nobody
    touched. Storing what the database can actually store removes the
    discrepancy rather than papering over it in the hash function.
    """
    return datetime.now(UTC).replace(tzinfo=None)


class ChainedRecord:
    """Mixin for a table in the hash chain.

    `prev_hash` is UNIQUE, which is what makes a fork impossible rather than
    merely unlikely: two rows cannot both claim the same predecessor, so a
    concurrent pair of inserts that read the same chain tail ends with one
    winner and one `IntegrityError` the writer retries
    (`disputedesk/audit/log.py`). Without it, two writers racing would produce
    two branches and `verify_chain` would report a break that nobody caused.

    `chain_payload()` is the row's contribution to its own hash. It is
    deliberately explicit rather than "every column": `id` is excluded (a
    database-assigned sequence, not a fact about the decision) and so are
    `prev_hash`/`row_hash` themselves (the hash cannot cover itself). Every
    remaining field - including `features_json`, which is what a reviewer
    reconstructs the decision from - is covered.
    """

    prev_hash: Mapped[str | None] = mapped_column(String, nullable=True, unique=True)
    row_hash: Mapped[str] = mapped_column(String, nullable=False)

    def chain_payload(self) -> dict:
        raise NotImplementedError


class DecisionRecord(ChainedRecord, Base):
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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_utc)

    def chain_payload(self) -> dict:
        return {
            "dispute_id": self.dispute_id,
            "reason_code": self.reason_code,
            "amount_inr": self.amount_inr,
            "model_version": self.model_version,
            "features_json": self.features_json,
            "p_win": self.p_win,
            "policy_branch": self.policy_branch,
            "expected_value_inr": self.expected_value_inr,
            "representment_cost_inr": self.representment_cost_inr,
            "low_confidence": self.low_confidence,
            "prompt_version": self.prompt_version,
            "validation_result": self.validation_result,
            "human_review_required": self.human_review_required,
            "created_at": self.created_at,
        }


class ApiOutcome(ChainedRecord, Base):
    __tablename__ = "api_outcomes"
    __table_args__ = (UniqueConstraint("dispute_id", name="uq_api_outcomes_dispute_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dispute_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    action: Mapped[str] = mapped_column(String, nullable=False)  # "accept" | "contest"
    # "success" | "failed" | "withheld_for_review" - the last meaning the
    # policy engine reached a decision but the packet was not fit to act on
    # unsupervised, so nothing was filed (see disputedesk/api/pipeline.py).
    outcome: Mapped[str] = mapped_column(String, nullable=False)
    response_json: Mapped[str | None] = mapped_column(String, nullable=True)
    error: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_utc)

    def chain_payload(self) -> dict:
        return {
            "dispute_id": self.dispute_id,
            "action": self.action,
            "outcome": self.outcome,
            "response_json": self.response_json,
            "error": self.error,
            "created_at": self.created_at,
        }


CHAINED_MODELS: tuple[type[ChainedRecord], ...] = (DecisionRecord, ApiOutcome)
"""Every table `disputedesk.audit.chain.verify_chain` walks. A new audit table
must be added here, or it is outside the chain and nothing will say so."""

APPEND_ONLY_TABLES: tuple[str, ...] = tuple(m.__tablename__ for m in CHAINED_MODELS)
"""Tables `disputedesk.audit.db` installs UPDATE/DELETE guards on."""
