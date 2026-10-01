"""
Measure cross-camera co-visibility and transitions within one camera group.

WHY
---
Before wiring cameras together (zone geometry for overlapping views, handoff
edges for sequential ones) we need to know, from the event log, which pairs
actually SEE the same people and in what temporal relationship:

  * CO-VISIBILITY  — one identity on two cameras at overlapping times.
    Evidence the views overlap -> ground-plane zone association applies.
  * TRANSITION     — one identity's visit on camera A ends, and a visit on
    camera B begins shortly after. Evidence of a walkable route -> a handoff
    edge, and the deltas seed its transit-time window.

Identity comes from person_id (enrolled people — reliable, independent per
camera) and unknown_person_id (strangers — today's cross-camera re-ID is ~0%,
so unknown-based numbers are a lower bound, not the truth).

Usage:
    python measure_coe_transitions.py                # default: cameras LIKE %coe%
    python measure_coe_transitions.py --like "front" # another group
    python measure_coe_transitions.py --days 7
"""

import argparse
from collections import defaultdict
from datetime import timedelta

from sqlalchemy import text

from app.db.session import SessionLocal

# A visit = consecutive events for one identity on one camera with gaps below
# this. Above it, the person left and came back.
VISIT_GAP_SEC = 60
# A->B counts as a transition only if B starts within this after A ends.
MAX_TRANSIT_SEC = 600


def load_sightings(db, like: str, days: int):
    """[(identity_key, camera_name, ts)] ordered by identity then time."""
    rows = db.execute(
        text("""
            SELECT
                CASE
                    WHEN e.person_id IS NOT NULL THEN 'known:' || e.person_id::text
                    WHEN e.unknown_person_id IS NOT NULL THEN 'unknown:' || e.unknown_person_id
                END AS identity,
                c.name AS camera,
                e.created_at AS ts
            FROM detection_events e
            JOIN cameras c ON c.id = e.camera_id
            WHERE LOWER(c.name) LIKE :like
              AND e.created_at > NOW() - make_interval(days => :days)
              AND (e.person_id IS NOT NULL OR e.unknown_person_id IS NOT NULL)
            ORDER BY identity, e.created_at
        """),
        {"like": f"%{like.lower()}%", "days": days},
    ).fetchall()
    return [(r.identity, r.camera, r.ts) for r in rows if r.identity]


def build_visits(sightings):
    """identity -> [(camera, start, end)], time-ordered, gap-collapsed."""
    visits = defaultdict(list)
    for identity, camera, ts in sightings:
        v = visits[identity]
        if v and v[-1][0] == camera and (ts - v[-1][2]).total_seconds() <= VISIT_GAP_SEC:
            v[-1] = (camera, v[-1][1], ts)
        else:
            v.append((camera, ts, ts))
    for v in visits.values():
        v.sort(key=lambda x: x[1])
    return visits


def analyse(visits):
    covis = defaultdict(int)        # frozenset({A,B}) -> count
    trans = defaultdict(list)       # (A,B) -> [delta_sec]
    multi_cam_ids = {"known": 0, "unknown": 0}
    single_cam_ids = {"known": 0, "unknown": 0}

    for identity, v in visits.items():
        kind = identity.split(":", 1)[0]
        cams = {c for c, _, _ in v}
        (multi_cam_ids if len(cams) > 1 else single_cam_ids)[kind] += 1

        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                (ca, sa, ea), (cb, sb, eb) = v[i], v[j]
                if ca == cb:
                    continue
                if sb <= ea:                       # time overlap
                    covis[frozenset((ca, cb))] += 1
                else:
                    delta = (sb - ea).total_seconds()
                    # only the NEXT visit counts as a transition; anything
                    # after an intervening visit isn't a direct walk.
                    if j == i + 1 and delta <= MAX_TRANSIT_SEC:
                        trans[(ca, cb)].append(delta)
    return covis, trans, multi_cam_ids, single_cam_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--like", default="coe", help="camera name filter (SQL LIKE, case-insensitive)")
    ap.add_argument("--days", type=int, default=30)
    args = ap.parse_args()

    db = SessionLocal()
    try:
        sightings = load_sightings(db, args.like, args.days)
    finally:
        db.close()

    print(f"{len(sightings)} identity sightings on cameras LIKE '%{args.like}%' "
          f"in the last {args.days} days")
    visits = build_visits(sightings)
    n_visits = sum(len(v) for v in visits.values())
    print(f"{len(visits)} distinct identities, {n_visits} visits "
          f"(gap {VISIT_GAP_SEC}s)\n")

    covis, trans, multi, single = analyse(visits)

    print("== identities seen on >1 camera (upper bound on linkability today) ==")
    for kind in ("known", "unknown"):
        total = multi[kind] + single[kind]
        pct = 100.0 * multi[kind] / total if total else 0.0
        print(f"  {kind:8s}: {multi[kind]}/{total}  ({pct:.1f}%)")

    print("\n== co-visibility (same identity, overlapping time) -> OVERLAP/zone pairs ==")
    if not covis:
        print("  none observed")
    for pair, n in sorted(covis.items(), key=lambda kv: -kv[1]):
        a, b = sorted(pair)
        print(f"  {a:22s} <-> {b:22s}  {n:4d} overlaps")

    print("\n== sequential transitions (A ends, B starts) -> HANDOFF edges ==")
    if not trans:
        print("  none observed")
    for (a, b), deltas in sorted(trans.items(), key=lambda kv: -len(kv[1])):
        deltas.sort()
        med = deltas[len(deltas) // 2]
        print(f"  {a:22s} --> {b:22s}  n={len(deltas):3d}  "
              f"transit s: min={deltas[0]:.0f} med={med:.0f} max={deltas[-1]:.0f}")


if __name__ == "__main__":
    main()
