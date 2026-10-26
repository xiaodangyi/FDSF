"""Database lifecycle and the first versioned SQLite schema migration."""

from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from sqlalchemy import inspect, text

from .db import DEFAULT_DATABASE, make_engine, make_session_factory
from .models import Base, SchemaVersion


SCHEMA_VERSION = 3


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
            with self.engine.begin() as connection:
                tables = inspect(connection).get_table_names()
                current = (connection.scalar(text('SELECT max(version) FROM schema_version')) or 0
                           if 'schema_version' in tables else 0)
                if current > SCHEMA_VERSION:
                    raise RuntimeError(
                        f'Database schema {current} is newer than supported {SCHEMA_VERSION}.')
                Base.metadata.create_all(connection)
                if 'owner_id' not in {col['name'] for col in inspect(connection).get_columns('runs')}:
                    connection.execute(text('ALTER TABLE runs ADD COLUMN owner_id VARCHAR(36) '
                                            'REFERENCES users(id)'))
                connection.execute(text('CREATE INDEX IF NOT EXISTS ix_runs_owner_id ON runs(owner_id)'))
                columns = {col['name'] for col in inspect(connection).get_columns('runs')}
                for name, kind in (('execution_id', 'VARCHAR(36)'),
                                   ('heartbeat_at', 'INTEGER'), ('started_routes', 'JSON')):
                    if name not in columns:
                        connection.execute(text(f'ALTER TABLE runs ADD COLUMN {name} {kind}'))
                if current < SCHEMA_VERSION:
                    connection.execute(SchemaVersion.__table__.insert().values(
                        version=SCHEMA_VERSION,
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
