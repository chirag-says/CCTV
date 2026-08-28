"""
Apply a hand-written SQL migration using the application's own DB settings.

WHY THIS EXISTS
There is no Alembic here: every migration from 002 onward is applied by hand,
and the documented way was a `psql` command. `psql` is not on PATH on a normal
Windows install (it lives under C:\\Program Files\\PostgreSQL\\<ver>\\bin), and
retyping host/user/password invites applying a migration to the wrong database.

This reads app.config.settings, so it always targets exactly the database the
backend itself talks to.

USAGE
    python apply_migration.py app/migrations/011_unknown_person_id.sql
    python apply_migration.py --list
    python apply_migration.py --check            # report schema drift only

The migration files carry their own BEGIN/COMMIT, so the connection runs in
autocommit and the file controls its own transaction. A file without BEGIN is
still atomic per statement, which is why every migration here opens one.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

MIGRATIONS_DIR = Path(__file__).parent / "app" / "migrations"


def _connect():
    import psycopg2
    from app.config import settings

    print(
        f"Target: {settings.DB_USER}@{settings.DB_HOST}:{settings.DB_PORT}"
        f"/{settings.DB_NAME}"
    )
    conn = psycopg2.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        dbname=settings.DB_NAME,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
    )
    # The .sql files manage their own transactions.
    conn.autocommit = True
    return conn


def list_migrations():
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        print(f"  {path.name}")


def check_schema():
    """Report model columns the live database is missing, and exit non-zero."""
    from app.db.session import verify_schema

    missing = verify_schema()
    if not missing:
        print("Schema is current: every model column exists.")
        return 0
    # verify_schema already logged the detail at CRITICAL.
    print("\nSchema is BEHIND the code. Apply the migration that adds:")
    for table, cols in sorted(missing.items()):
        print(f"  {table}: {', '.join(sorted(cols))}")
    return 1


def apply(path: Path) -> int:
    if not path.exists():
        print(f"No such migration: {path}")
        return 1

    sql = path.read_text(encoding="utf-8")
    conn = _connect()
    try:
        with conn.cursor() as cur:
            # psycopg2 sends the whole script in one round trip, so the DO
            # block and its dollar-quoting survive intact. Splitting on ';'
            # would tear that apart.
            cur.execute(sql)
            # RAISE NOTICE output (e.g. the backfill row count) arrives here.
            for notice in conn.notices:
                print("  " + notice.strip())
        print(f"\nApplied {path.name}")
        return 0
    except Exception as e:
        print(f"\nFAILED: {path.name}\n  {e}")
        return 1
    finally:
        conn.close()


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if args[0] == "--list":
        list_migrations()
        return 0
    if args[0] == "--check":
        return check_schema()

    target = Path(args[0])
    if not target.exists():
        candidate = MIGRATIONS_DIR / target.name
        if candidate.exists():
            target = candidate
    return apply(target)


if __name__ == "__main__":
    raise SystemExit(main())
