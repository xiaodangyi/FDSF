"""SQLite engine and session configuration for the S3 history store."""

from pathlib import Path
import os

from sqlalchemy import URL, create_engine, event
from sqlalchemy.orm import sessionmaker

from .config import ROOT


DEFAULT_DATABASE = Path(os.environ.get('DESKVISION_DATABASE', ROOT / 'runs/stage3.sqlite3'))


def make_engine(path: str | Path = DEFAULT_DATABASE):
    database_path = Path(path).expanduser().resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        URL.create('sqlite', database=str(database_path)),
        connect_args={'check_same_thread': False, 'timeout': 30},
    )

    @event.listens_for(engine, 'connect')
    def configure_sqlite(connection, _record):
        cursor = connection.cursor()
        cursor.execute('PRAGMA foreign_keys=ON')
        cursor.execute('PRAGMA busy_timeout=30000')
        cursor.execute('PRAGMA journal_mode=WAL')
        cursor.close()

    return engine


def make_session_factory(engine):
    return sessionmaker(bind=engine, expire_on_commit=False)
