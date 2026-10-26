"""Initialize or upgrade the local SQLite database."""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deskvision.database import Database, SCHEMA_VERSION


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', help='Database file path; defaults to runs/stage3.sqlite3.')
    args = parser.parse_args()
    database = Database(args.database) if args.database else Database()
    database.initialize()
    print(f'Initialized SQLite schema version {SCHEMA_VERSION} at {database.path}')
    database.dispose()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
