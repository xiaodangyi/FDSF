"""SQLite S3 schema and restart durability checks."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from datetime import datetime

from sqlalchemy import select

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
