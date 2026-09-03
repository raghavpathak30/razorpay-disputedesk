"""SQLite-via-SQLAlchemy engine/session setup, plus the database-level
append-only guards on the audit tables.

A correction to this module's previous docstring, which claimed "nothing here
is SQLite-specific except the `check_same_thread` connect arg". That was true
until 2026-09-02, when `init_db` gained the `BEFORE UPDATE`/`BEFORE DELETE`
triggers that make the audit log append-only in the store rather than only by
convention (remediation defect 0.4). Trigger DDL is dialect-specific, and this
project has written and tested only the SQLite form. `install_append_only_guards`
therefore *raises* on any other dialect rather than silently leaving a Postgres
deployment with an audit log that claims to be append-only and is not - which
is the exact failure this defect was. Porting to Postgres is now a
connection-string change **plus** the equivalent `CREATE TRIGGER ... EXECUTE
FUNCTION` DDL (or a `REVOKE UPDATE, DELETE` on the application role, which is
the cleaner Postgres answer); see ARCHITECTURE.md's "Known gaps".
"""

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from disputedesk.audit.models import APPEND_ONLY_TABLES, Base
from disputedesk.config import get_settings


class AppendOnlyGuardsUnavailableError(RuntimeError):
    """Raised when the configured database dialect has no tested append-only
    guard DDL in this project. Fails closed: an audit log that cannot be made
    append-only must not quietly come up as if it were.
    """


def get_engine(database_url: str | None = None) -> Engine:
    """`database_url` defaults to `get_settings().database_url`; callers
    (tests, the demo script) that want an isolated database pass one
    explicitly, e.g. `"sqlite:///:memory:"` or a temp-file URL, without
    needing the rest of `Settings` populated.
    """
    url = database_url or get_settings().database_url
    if not url.startswith("sqlite"):
        return create_engine(url)

    # `:memory:` SQLite is per-connection: without a shared connection, each
    # checkout from the pool would see a *different*, empty database.
    # `StaticPool` keeps the whole engine on one connection so callers that
    # pass `"sqlite:///:memory:"` (tests, the demo script) see one
    # consistent database across every session, as `get_engine`'s docstring
    # promises.
    is_memory = url in ("sqlite:///:memory:", "sqlite://")
    kwargs = {"connect_args": {"check_same_thread": False}}
    if is_memory:
        kwargs["poolclass"] = StaticPool
    return create_engine(url, **kwargs)


def install_append_only_guards(engine: Engine) -> None:
    """Install a `BEFORE UPDATE` and a `BEFORE DELETE` trigger on every audit
    table, each aborting the statement.

    Idempotent (`IF NOT EXISTS`), so it is safe on every process start, the
    same as `create_all`. The message is written to be read by whoever hits it
    in a stack trace, since by construction that person is doing something the
    system says is not allowed.
    """
    if engine.dialect.name != "sqlite":
        raise AppendOnlyGuardsUnavailableError(
            f"append-only trigger DDL is implemented and tested for sqlite only, "
            f"not {engine.dialect.name!r}. See this module's docstring: a "
            f"deployment on another dialect must install the equivalent guards "
            f"(or REVOKE UPDATE, DELETE on the application role) before the "
            f"audit log can be described as append-only."
        )

    with engine.begin() as connection:
        for table in APPEND_ONLY_TABLES:
            for verb in ("UPDATE", "DELETE"):
                connection.execute(
                    text(
                        f"CREATE TRIGGER IF NOT EXISTS {table}_no_{verb.lower()} "
                        f"BEFORE {verb} ON {table} "
                        f"BEGIN SELECT RAISE(ABORT, "
                        f"'{table} is append-only: {verb} is not permitted'); END"
                    )
                )


def init_db(engine: Engine) -> None:
    """Create every table if it does not already exist, then install the
    append-only guards. Safe to call every process startup - both steps are
    no-ops when they have already run.
    """
    Base.metadata.create_all(engine)
    install_append_only_guards(engine)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)
