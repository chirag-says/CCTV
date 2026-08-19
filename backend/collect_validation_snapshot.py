"""
Capture one point-in-time snapshot of system state for validation review.

Run this before starting a long validation run and again after it finishes. The
difference between two snapshots is the evidence: how many unknown identities
were created, how many were duplicates, whether the registry stayed bounded,
whether the gallery changed.

Raw logs alone do not answer those questions -- they show events, not
aggregates, and they cannot be diffed. This writes a timestamped directory:

    validation/<timestamp>/
        summary.txt          human-readable, read this first
        snapshot.json        machine-readable, diff two of these
        gallery.txt          full face-gallery diagnostics report
        gallery.json         same, machine-readable
        performance.log      copy of the profiler log as of this moment

Usage:
    python collect_validation_snapshot.py                  # before the run
    python collect_validation_snapshot.py --label after    # after the run
    python collect_validation_snapshot.py --compare validation/A validation/B
"""

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, ".")

VALIDATION_DIR = "validation"


def _collect() -> dict:
    """Everything worth knowing about the system right now."""
    from sqlalchemy import func

    from app.db.session import SessionLocal
    from app.db.models import (
        DetectionEvent, FaceEncoding, Person, UnknownFace,
    )

    db = SessionLocal()
    try:
        now = datetime.now(timezone.utc)
        day_ago = now - timedelta(days=1)

        unknown_by_status = {
            (s or "unknown"): int(c)
            for s, c in db.query(UnknownFace.status, func.count(UnknownFace.id))
                          .group_by(UnknownFace.status).all()
        }
        total_unknown = sum(unknown_by_status.values())
        with_encoding = db.query(UnknownFace).filter(
            UnknownFace.face_encoding.isnot(None)
        ).count()

        # Occurrence histogram. This is the duplicate-rate signal: if one real
        # person maps to one identity, repeat visitors accumulate occurrences.
        # A table dominated by occurrence=1 means dedup is not working.
        occ_hist = {
            int(o or 0): int(c)
            for o, c in db.query(UnknownFace.occurrence, func.count(UnknownFace.id))
                          .filter(UnknownFace.status == "pending")
                          .group_by(UnknownFace.occurrence).all()
        }
        pending = unknown_by_status.get("pending", 0)
        singletons = occ_hist.get(1, 0)

        events_by_type = {
            (t or "unknown"): int(c)
            for t, c in db.query(DetectionEvent.event_type, func.count(DetectionEvent.id))
                          .group_by(DetectionEvent.event_type).all()
        }

        return {
            "captured_at": now.isoformat(),
            "persons": {
                "total": db.query(Person).count(),
                "active": db.query(Person).filter(Person.is_active == True).count(),
                "encodings": db.query(FaceEncoding).count(),
            },
            "unknown_faces": {
                "total": total_unknown,
                "by_status": unknown_by_status,
                "with_encoding": with_encoding,
                "without_encoding": total_unknown - with_encoding,
                "pending_occurrence_1": singletons,
                "pending_singleton_rate": (singletons / pending) if pending else None,
                "occurrence_histogram": dict(sorted(occ_hist.items())[:20]),
                "created_last_24h": db.query(UnknownFace).filter(
                    UnknownFace.created_at >= day_ago
                ).count(),
            },
            "detection_events": {
                "total": db.query(DetectionEvent).count(),
                "by_type": events_by_type,
                "last_24h": db.query(DetectionEvent).filter(
                    DetectionEvent.created_at >= day_ago
                ).count(),
            },
        }
    finally:
        db.close()


def _live_camera_status(token: str = "") -> dict:
    """
    Runtime counters, if the API server is reachable.

    The endpoint requires authentication, so without a token this returns
    "unavailable" and the registry counters -- resident identities, views
    learned, convergences -- are missing from the snapshot. Pass --token (or set
    SENTINEL_TOKEN) when capturing against a running system.

    Deliberately optional either way: the snapshot must still be useful when run
    against a stopped system, which is the normal case for the "after" capture.
    """
    try:
        import urllib.request

        from app.config import settings

        token = token or os.environ.get("SENTINEL_TOKEN", "")
        url = f"http://127.0.0.1:{settings.PORT}/api/cameras/status"
        request = urllib.request.Request(url)
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(request, timeout=3) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        hint = "" if token else " (no --token given)"
        return {"unavailable": f"{e}{hint}"}


def _format(snapshot: dict, gallery_line: str) -> str:
    u = snapshot["unknown_faces"]
    p = snapshot["persons"]
    e = snapshot["detection_events"]
    out = [
        "=" * 78,
        "SENTINELAI VALIDATION SNAPSHOT",
        "=" * 78,
        f"Captured: {snapshot['captured_at']}",
        "",
        "ENROLLED",
        f"  Persons (active): {p['active']} of {p['total']}",
        f"  Face encodings:   {p['encodings']}",
        f"  Gallery health:   {gallery_line}",
        "",
        "UNKNOWN IDENTITIES",
        f"  Total rows:       {u['total']}",
        f"  By status:        " + ", ".join(f"{k}={v}" for k, v in sorted(u["by_status"].items())),
        f"  Without encoding: {u['without_encoding']}  (cannot be deduped or enrolled)",
        f"  Created in 24h:   {u['created_last_24h']}",
    ]
    rate = u["pending_singleton_rate"]
    if rate is not None:
        verdict = "GOOD" if rate < 0.5 else ("POOR" if rate > 0.8 else "MARGINAL")
        out.append(
            f"  Seen exactly once: {u['pending_occurrence_1']} of "
            f"{u['by_status'].get('pending', 0)} pending ({rate * 100:.1f}%)  [{verdict}]"
        )
        out.append(
            "      A high rate means one real person is producing many identities."
        )
    out += [
        "",
        "EVENTS",
        f"  Total:            {e['total']}",
        f"  Last 24h:         {e['last_24h']}",
        f"  By type:          " + ", ".join(f"{k}={v}" for k, v in sorted(e["by_type"].items())),
        "",
        "CAMERAS (live)",
    ]
    cameras = snapshot.get("cameras", {})
    if cameras.get("unavailable"):
        out.append(f"  Not reachable: {cameras['unavailable']}")
    else:
        for cid, status in (cameras.get("cameras") or {}).items():
            reg = status.get("unknown_registry") or {}
            out.append(
                f"  {cid}: fps={status.get('fps')} frames={status.get('frames_processed')} "
                f"faces_cached={status.get('faces_in_cache')}"
            )
            if reg:
                out.append(
                    f"      registry: resident={reg.get('resident')} views={reg.get('views')} "
                    f"minted={reg.get('minted')} matched={reg.get('matched')} "
                    f"learned={reg.get('views_learned')} converged={reg.get('converged')} "
                    f"evicted={reg.get('evicted')}"
                )
    out.append("=" * 78)
    return "\n".join(out)


def _compare(before_dir: str, after_dir: str) -> str:
    def load(d):
        with open(os.path.join(d, "snapshot.json"), encoding="utf-8") as fh:
            return json.load(fh)

    a, b = load(before_dir), load(after_dir)
    au, bu = a["unknown_faces"], b["unknown_faces"]
    ae, be = a["detection_events"], b["detection_events"]

    def delta(x, y):
        return f"{x} -> {y}  ({y - x:+d})"

    lines = [
        "=" * 78,
        "VALIDATION RUN COMPARISON",
        "=" * 78,
        f"Before: {a['captured_at']}",
        f"After:  {b['captured_at']}",
        "",
        f"  Unknown rows:      {delta(au['total'], bu['total'])}",
        f"  Pending:           {delta(au['by_status'].get('pending', 0), bu['by_status'].get('pending', 0))}",
        f"  Without encoding:  {delta(au['without_encoding'], bu['without_encoding'])}",
        f"  Detection events:  {delta(ae['total'], be['total'])}",
        f"  Face encodings:    {delta(a['persons']['encodings'], b['persons']['encodings'])}",
    ]
    for label, snap in (("Before", au), ("After", bu)):
        rate = snap["pending_singleton_rate"]
        if rate is not None:
            lines.append(f"  Singleton rate ({label.lower()}): {rate * 100:.1f}%")
    lines += [
        "",
        "  A run that adds many unknown rows with a high singleton rate is still",
        "  producing duplicate identities. Compare gallery.txt in both directories",
        "  to see whether enrollment changed underneath the numbers.",
        "=" * 78,
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture a validation snapshot.")
    parser.add_argument("--label", default="", help="suffix for the output directory")
    parser.add_argument(
        "--token", default="",
        help="bearer token for live camera counters (or set SENTINEL_TOKEN)",
    )
    parser.add_argument(
        "--compare", nargs=2, metavar=("BEFORE", "AFTER"),
        help="compare two existing snapshot directories and exit",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    if args.compare:
        print(_compare(*args.compare))
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_{args.label}" if args.label else stamp
    out_dir = os.path.join(VALIDATION_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    try:
        snapshot = _collect()
    except Exception as e:
        print(f"Could not read the database: {e}", file=sys.stderr)
        return 2
    snapshot["cameras"] = _live_camera_status(args.token)

    # Gallery diagnostics, captured alongside so the two are always in sync.
    gallery_line = "not available"
    try:
        from app.services.person_service import PersonService
        from app.vision.gallery_diagnostics import (
            analyse_gallery, format_report, report_to_dict,
        )

        report = analyse_gallery(PersonService.get_all_encodings())
        with open(os.path.join(out_dir, "gallery.txt"), "w", encoding="utf-8") as fh:
            fh.write(format_report(report, verbose=True) + "\n")
        payload = report_to_dict(report)
        with open(os.path.join(out_dir, "gallery.json"), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        snapshot["gallery"] = payload["summary"]
        rec = payload["summary"]["recognise_rate"]
        gallery_line = (
            f"{payload['summary']['high_risk_persons']} high-risk persons, "
            f"{payload['summary']['split_identities']} split identities, "
            + (f"{rec * 100:.1f}% LOO recognition" if rec is not None else "not measurable")
        )
    except Exception as e:
        snapshot["gallery"] = {"error": str(e)}

    perf_log = os.path.join("logs", "performance.log")
    if os.path.exists(perf_log):
        shutil.copy2(perf_log, os.path.join(out_dir, "performance.log"))

    with open(os.path.join(out_dir, "snapshot.json"), "w", encoding="utf-8") as fh:
        json.dump(snapshot, fh, indent=2)

    summary = _format(snapshot, gallery_line)
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(summary + "\n")

    print(summary)
    print(f"\nWritten to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
