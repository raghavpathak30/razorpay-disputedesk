"""SQLite-via-SQLAlchemy engine/session setup (CLAUDE.md: "so Postgres is a
connection-string change, not a rewrite" - nothing here is SQLite-specific
except the `check_same_thread` connect arg, itself only applied when the URL
is a `sqlite:` one).
"""

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from disputedesk.audit.models import Base
from disputedesk.config import get_settings


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


def init_db(engine: Engine) -> None:
    """Create every table if it does not already exist. Safe to call every
    process startup - `create_all` is a no-op for tables that already exist.
    """
    Base.metadata.create_all(engine)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)
