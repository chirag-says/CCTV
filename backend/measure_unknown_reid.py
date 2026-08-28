"""
Measure re-identification of unenrolled people.

This is the number Step 3.5 exists to move. Before that work it was:

    cross-camera re-ID  0.0%   (415 strangers, 7 live cameras, none matched
                                on two of them)
    cross-day  re-ID    0.5%
    views per identity  1.00   (best-of-K silently degraded to best-of-1)

Run it after a real camera session. Reads only; writes nothing.

USAGE
    python measure_unknown_reid.py              # last 7 days, plus per-day
    python measure_unknown_reid.py --days 1     # just today
    python measure_unknown_reid.py --since 2026-08-28T14:00

WHAT TO LOOK AT, IN ORDER
1. `views per identity` > 1.00. If it stays at exactly 1.00 the registry is
   not persisting what it learns, and nothing below can improve.
2. `cross-camera` off 0.0%. This is the headline.
3. `sightings per stranger`. A spike at exactly 2 means one entry/exit pair
   each -- every stranger is a first-time visitor, i.e. write-time dedup is
   still minting a fresh identity per visit. That is a DIFFERENT problem from
   the one Step 3.5 fixed (see UNKNOWN_SIMILARITY_THRESHOLD), and no amount of
   multi-view persistence will move it.
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def _connect():
    import psycopg2
    from app.config import settings

    return psycopg2.connect(
        host=settings.DB_HOST, port=settings.DB_PORT, dbname=settings.DB_NAME,
        user=settings.DB_USER, password=settings.DB_PASSWORD,
    )


def _pct(part, whole):
    return f"{part / whole:.1%}" if whole else "n/a"


def report(since: datetime):
    conn = _connect()
    cur = conn.cursor()
    print(f"Window: since {since.isoformat(sep=' ', timespec='minutes')}\n")

    # ── Views per identity: does persistence work at all? ──
    cur.execute("""
        SELECT COALESCE(AVG(views), 0), COALESCE(MAX(views), 0),
               COUNT(*) FILTER (WHERE views > 1), COUNT(*)
          FROM (SELECT unknown_face_id, COUNT(*) AS views
                  FROM unknown_face_views GROUP BY 1) t""")
    avg_v, max_v, multi, total_v = cur.fetchone()
    print("MULTI-VIEW PERSISTENCE")
    print(f"  identities with views : {total_v:,}")
    print(f"  views per identity    : {avg_v:.2f} avg, {max_v} max")
    print(f"  with more than one    : {multi:,} ({_pct(multi, total_v)})")
    if total_v and multi == 0:
        print("  !! every identity still has exactly one view -- the registry")
        print("     is not persisting learned views; fix this before reading on")

    # ── Re-identification ──
    cur.execute("""
        WITH s AS (
          SELECT unknown_person_id AS uid,
                 COUNT(*) AS events,
                 COUNT(DISTINCT created_at::date) AS days,
                 COUNT(DISTINCT camera_id) AS cams
            FROM detection_events
           WHERE unknown_person_id IS NOT NULL AND created_at >= %s
           GROUP BY 1)
        SELECT COUNT(*), COUNT(*) FILTER (WHERE cams > 1),
               COUNT(*) FILTER (WHERE days > 1), COUNT(*) FILTER (WHERE events > 2)
          FROM s""", (since,))
    n, multicam, multiday, revisit = cur.fetchone()

    cur.execute("""SELECT COUNT(DISTINCT camera_id) FROM detection_events
                    WHERE created_at >= %s""", (since,))
    cameras = cur.fetchone()[0]

    print(f"\nRE-IDENTIFICATION  ({n:,} strangers, {cameras} camera(s) active)")
    if not n:
        print("  no unknown-identity events in this window")
        conn.close()
        return
    print(f"  cross-camera : {_pct(multicam, n):>6}  ({multicam:,})   was 0.0%")
    print(f"  cross-day    : {_pct(multiday, n):>6}  ({multiday:,})   was 0.5%")
    print(f"  revisited    : {_pct(revisit, n):>6}  ({revisit:,})")
    if cameras < 2:
        print("  (cross-camera is meaningless with one camera active)")

    # ── Sightings distribution: is write-time dedup converging? ──
    cur.execute("""
        SELECT events, COUNT(*) FROM (
          SELECT unknown_person_id, COUNT(*) AS events
            FROM detection_events
           WHERE unknown_person_id IS NOT NULL AND created_at >= %s
           GROUP BY 1) t
        GROUP BY 1 ORDER BY 1 LIMIT 8""", (since,))
    print("\nSIGHTINGS PER STRANGER  (2 = a single entry/exit pair)")
    for events, people in cur.fetchall():
        bar = "#" * min(40, int(40 * people / n))
        print(f"  {events:>3} : {people:>6,}  {bar}")

    conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--since", type=str, default=None,
                    help="ISO timestamp; overrides --days")
    args = ap.parse_args()

    if args.since:
        since = datetime.fromisoformat(args.since)
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)
    else:
        since = datetime.now(timezone.utc) - timedelta(days=args.days)

    report(since)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
