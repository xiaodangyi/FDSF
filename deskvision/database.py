"""Database lifecycle and the first versioned SQLite schema migration."""

from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from sqlalchemy import func, select

from .db import DEFAULT_DATABASE, make_engine, make_session_factory
from .models import Base, SchemaVersion


SCHEMA_VERSION = 1


class Database:
    def __init__(self, path: str | Path = DEFAULT_DATABASE):
        self.path = Path(path).expanduser().resolve()
        self.engine = make_engine(self.path)
        self.sessions = make_session_factory(self.engine)
        self._lock = Lock()
        self._initialized = False

    def initialize(self) -> None:
        """Apply the idempotent initial migration and reject unknown future schemas."""
        with self._lock:
            if self._initialized:
                return
            Base.metadata.create_all(self.engine)
            with self.sessions.begin() as session:
                current = session.scalar(select(func.max(SchemaVersion.version))) or 0
                if current > SCHEMA_VERSION:
                    raise RuntimeError(
                        f'Database schema {current} is newer than supported {SCHEMA_VERSION}.')
                if current == 0:
                    session.add(SchemaVersion(version=SCHEMA_VERSION,
                                              applied_at=datetime.now(timezone.utc).replace(tzinfo=None)))
            self._initialized = True

    def dispose(self) -> None:
        self.engine.dispose()


_database = Database()


def get_database() -> Database:
    return _database


def configure_database(path: str | Path) -> Database:
    """Replace the database target; primarily used for isolated tests."""
    global _database
    previous = _database
    _database = Database(path)
    previous.dispose()
    return _database
