"""
Backfill real pose metadata onto unknown_faces rows written before migration 010.

Rows created before the pose fix have NULL yaw/pitch/roll/pose_bucket, which
makes them ineligible for the Gallery Optimizer's diversity gate — by design,
since inventing a pose would corrupt every coverage number downstream.

This script recovers the pose honestly: it re-runs SCRFD over each row's stored
face crop, takes the 5-point landmarks from that detection, and computes the
same yaw/pitch/roll the live pipeline now computes at capture time. Rows whose
snapshot is missing, unreadable, or yields no detection are left NULL.

Usage:
    python backfill_unknown_pose.py --dry-run            # report only
    python backfill_unknown_pose.py --limit 200          # do 200 rows
    python backfill_unknown_pose.py                      # do all pending rows

Safe to interrupt and re-run: it only touches rows that still have no pose.
"""

import argparse
import logging
import os
import sys
import time
from collections import Counter

import cv2

logging.basicConfig(level=logging.WARNING, format="%(message)s")

from app.db.session import SessionLocal
from app.db.models import UnknownFace
from app.services.embedding_suggestions import classify_pose_bucket
from app.vision.detector import FaceDetector
from app.vision.quality import FaceQualityGate


def snapshot_path(url: str) -> str:
    """Map a stored snapshot URL (/snapshots/...) to a path on disk."""
    return url.lstrip("/").replace("/", os.sep)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="max rows to process (0 = all)")
    ap.add_argument("--dry-run", action="store_true", help="compute but do not write")
    ap.add_argument("--status", default="pending", help="unknown_faces.status to process")
    ap.add_argument("--commit-every", type=int, default=200)
    args = ap.parse_args()

    db = SessionLocal()
    detector = FaceDetector()
    gate = FaceQualityGate()

    q = (
        db.query(UnknownFace)
        .filter(
            UnknownFace.status == args.status,
            UnknownFace.pose_bucket.is_(None),
            UnknownFace.snapshot_url.isnot(None),
        )
        .order_by(UnknownFace.last_seen.desc())
    )
    total = q.count()
    if args.limit:
        q = q.limit(args.limit)
    rows = q.all()

    print(f"{total} row(s) without pose; processing {len(rows)}"
          f"{' (dry run)' if args.dry_run else ''}\n")

    stats = Counter()
    buckets = Counter()
    started = time.time()

    for i, face in enumerate(rows, 1):
        path = snapshot_path(face.snapshot_url)
        if not os.path.exists(path):
            stats["missing_file"] += 1
            continue

        img = cv2.imread(path)
        if img is None:
            stats["unreadable"] += 1
            continue

        # Re-detect on the stored crop to recover the landmarks. The crop was
        # saved with padding, so SCRFD has enough context to localise again.
        detections = detector.detect_faces_raw(img)
        if not detections:
            stats["no_detection"] += 1
            continue

        # Largest detection — the crop is centred on one face.
        best = max(detections, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        kps = getattr(best, "kps", None)
        if kps is None or len(kps) < 5:
            stats["no_landmarks"] += 1
            continue

        x1, y1, x2, y2 = best.bbox.astype(int)
        location = (y1, x2, y2, x1)
        crop = img[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            stats["bad_bbox"] += 1
            continue

        result = gate.assess(
            face_crop=crop,
            face_location=location,
            det_score=float(getattr(best, "det_score", 1.0) or 1.0),
            landmarks=kps,
        )
        m = result.metrics
        yaw = float(m.get("yaw", 0.0))

        if not args.dry_run:
            face.yaw = yaw
            face.pitch = float(m.get("pitch", 0.0))
            face.roll = float(m.get("roll", 0.0))
            face.pose_bucket = classify_pose_bucket(yaw)
            face.quality = float(result.quality_score)
            face.blur_score = float(m.get("blur_score", 0.0))
            face.det_score = float(m.get("det_score", 0.0))
            face.lighting = m.get("lighting")

        buckets[classify_pose_bucket(yaw)] += 1
        stats["recovered"] += 1

        if not args.dry_run and i % args.commit_every == 0:
            db.commit()
            print(f"  ... {i}/{len(rows)} processed, {stats['recovered']} recovered")

    if not args.dry_run:
        db.commit()
    db.close()

    elapsed = time.time() - started
    print(f"\nProcessed {len(rows)} row(s) in {elapsed:.1f}s")
    print(f"  recovered      : {stats['recovered']}")
    for reason in ("missing_file", "unreadable", "no_detection", "no_landmarks", "bad_bbox"):
        if stats[reason]:
            print(f"  {reason:<15}: {stats[reason]}")
    print("\nPose distribution of recovered rows:")
    for bucket, n in buckets.most_common():
        pct = 100.0 * n / max(stats["recovered"], 1)
        print(f"  {bucket:<15} {n:>6}  {pct:5.1f}%")
    if args.dry_run:
        print("\n(dry run — nothing written)")


if __name__ == "__main__":
    sys.exit(main())
