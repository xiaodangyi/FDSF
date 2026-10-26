"""SQLite S3 schema and restart durability checks."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from datetime import datetime

from sqlalchemy import select, text

from deskvision.database import Database, SCHEMA_VERSION
from deskvision.models import SchemaVersion


class DatabaseTests(unittest.TestCase):
    def test_initial_migration_is_idempotent_and_records_schema_version(self):
        with TemporaryDirectory() as folder:
            database = Database(Path(folder) / 'history.sqlite3')
            database.initialize()
            database.initialize()
            with database.sessions() as session:
                versions = session.scalars(select(SchemaVersion.version)).all()
            self.assertEqual(versions, [SCHEMA_VERSION])
            self.assertTrue(database.path.is_file())
            database.dispose()

    def test_upgrade_from_s3_schema_keeps_existing_tasks_unowned(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / 's3.sqlite3'
            database = Database(path)
            with database.engine.begin() as connection:
                connection.execute(text('CREATE TABLE schema_version '
                                        '(version INTEGER PRIMARY KEY, applied_at DATETIME NOT NULL)'))
                connection.execute(text("INSERT INTO schema_version VALUES (1, '2026-10-26 00:00:00')"))
                connection.execute(text('''CREATE TABLE runs (
                    run_id VARCHAR(36) PRIMARY KEY, route VARCHAR(16) NOT NULL,
                    status VARCHAR(24) NOT NULL, created_at VARCHAR(40) NOT NULL,
                    started_at VARCHAR(40), finished_at VARCHAR(40), input_sha256 VARCHAR(64) NOT NULL,
                    mime_type VARCHAR(64) NOT NULL, input_bytes INTEGER NOT NULL,
                    input_width INTEGER NOT NULL, input_height INTEGER NOT NULL,
                    parameters JSON NOT NULL, input_path TEXT NOT NULL, run_dir TEXT NOT NULL,
                    comparison TEXT, error JSON, note VARCHAR(1000) NOT NULL DEFAULT '')'''))
                connection.execute(text("INSERT INTO runs (run_id, route, status, created_at, "
                                        "input_sha256, mime_type, input_bytes, input_width, "
                                        "input_height, parameters, input_path, run_dir) "
                                        "VALUES ('legacy-id', 'yolo', 'succeeded', '2026', 'hash', "
                                        "'image/png', 4, 1, 1, '{}', 'runs/stage3-api/legacy/input.img', "
                                        "'runs/stage3-api/legacy')"))
            database.initialize()
            with database.sessions() as session:
                self.assertIsNone(session.scalar(text("SELECT owner_id FROM runs WHERE run_id='legacy-id'")))
                self.assertEqual(session.scalars(select(SchemaVersion.version).order_by(
                    SchemaVersion.version)).all(), [1, SCHEMA_VERSION])
            database.dispose()

    def test_database_refuses_a_schema_version_it_cannot_read(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'future.sqlite3'
            database = Database(path)
            database.initialize()
            with database.sessions.begin() as session:
                session.add(SchemaVersion(version=SCHEMA_VERSION + 1,
                                          applied_at=datetime.now()))
            database.dispose()
            newer_database = Database(path)
            with self.assertRaisesRegex(RuntimeError, 'newer than supported'):
                newer_database.initialize()
            newer_database.dispose()


if __name__ == '__main__':
    unittest.main()
