"""
Gallery Optimizer backlog diagnostic — measure before you enable.

Answers, without writing a single row and without making a single VLM call:

  * how many unknown faces are pending, and how many are actually usable
  * how they cluster at the configured threshold, and how big those clusters are
  * how many members each cluster would flag as outliers
  * how similar each cluster is to its nearest enrolled person
  * what the pose coverage of the backlog looks like
  * how many VLM calls a real cycle would spend, and how many review
    candidates would land in the queue

Run this before flipping ENABLE_IDENTITY_REVIEW. The point is to see cluster
quality first: a bad clustering threshold turns into thousands of wrong review
candidates and thousands of wasted Qwen calls.

Usage:
    python gallery_optimizer_dryrun.py
    python gallery_optimizer_dryrun.py --limit 4000
    python gallery_optimizer_dryrun.py --threshold 0.45   # try another cut
"""

import argparse
import logging
import sys
from collections import Counter

import numpy as np

logging.basicConfig(level=logging.ERROR, format="%(message)s")

from app.config import settings
from app.db.session import SessionLocal
from app.db.models import FaceEncoding, Person, UnknownFace
from app.services.clustering_service import (
    centroid_distances, compute_cluster_groups, outlier_flags,
)
from app.services.identity_review_worker import (
    IdentityReviewWorker, compute_priority_score,
)
from app.vision.recognizer import deserialize_encoding


BAR = "=" * 62


def h(title):
    print(f"\n{BAR}\n  {title}\n{BAR}")


def histogram(counter, label_width=16, total=None):
    total = total or sum(counter.values())
    for key, n in sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        pct = 100.0 * n / max(total, 1)
        bar = "#" * int(pct / 2)
        print(f"  {str(key):<{label_width}} {n:>6}  {pct:5.1f}%  {bar}")


def bucket_size(n):
    if n < 3:
        return "1-2 (singleton)"
    if n <= 5:
        return "3-5"
    if n <= 10:
        return "6-10"
    if n <= 20:
        return "11-20"
    if n <= 50:
        return "21-50"
    return "51+"


def similarity_band(s):
    edges = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 0.98, 1.01]
    for lo, hi in zip(edges, edges[1:]):
        if lo <= s < hi:
            return f"{lo:.2f}-{hi:.2f}"
    return "?"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=2000,
                    help="max pending faces to load (matches the worker's own cap)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override IDENTITY_REVIEW_CLUSTER_THRESHOLD")
    ap.add_argument("--min-cluster-size", type=int, default=None)
    args = ap.parse_args()

    threshold = args.threshold if args.threshold is not None else settings.IDENTITY_REVIEW_CLUSTER_THRESHOLD
    min_size = args.min_cluster_size or settings.IDENTITY_REVIEW_MIN_CLUSTER_SIZE

    db = SessionLocal()
    worker = IdentityReviewWorker()

    # ── 1. Backlog ───────────────────────────────────────────────
    h("1. UNKNOWN BACKLOG")
    status_counts = Counter(
        dict(
            db.query(UnknownFace.status, __import__("sqlalchemy").func.count())
            .group_by(UnknownFace.status).all()
        )
    )
    total_unknowns = sum(status_counts.values())
    print(f"  total unknown_faces        : {total_unknowns}")
    histogram(status_counts, total=total_unknowns)

    pending_total = (
        db.query(UnknownFace).filter(UnknownFace.status == "pending").count()
    )
    with_enc = (
        db.query(UnknownFace)
        .filter(UnknownFace.status == "pending", UnknownFace.face_encoding.isnot(None))
        .count()
    )
    with_pose = (
        db.query(UnknownFace)
        .filter(
            UnknownFace.status == "pending",
            UnknownFace.face_encoding.isnot(None),
            UnknownFace.pose_bucket.isnot(None),
        )
        .count()
    )
    print(f"\n  pending                    : {pending_total}")
    print(f"  pending with embedding     : {with_enc}"
          f"   ({100.0 * with_enc / max(pending_total, 1):.1f}%)")
    print(f"  pending with REAL pose     : {with_pose}"
          f"   ({100.0 * with_pose / max(pending_total, 1):.1f}%)")
    print(f"  -> gallery-eligible today  : {with_pose}")
    if with_pose < with_enc:
        print(f"     {with_enc - with_pose} face(s) have an embedding but no pose;"
              f" they can be clustered\n     and reviewed, but cannot enter a gallery."
              f" Run backfill_unknown_pose.py.")

    # ── 2. Pose distribution ─────────────────────────────────────
    h("2. POSE DISTRIBUTION (pending faces)")
    pose_counts = Counter(
        dict(
            db.query(UnknownFace.pose_bucket, __import__("sqlalchemy").func.count())
            .filter(UnknownFace.status == "pending")
            .group_by(UnknownFace.pose_bucket).all()
        )
    )
    pose_counts = Counter({(k or "NULL (no pose)"): v for k, v in pose_counts.items()})
    histogram(pose_counts, total=pending_total)

    # ── 3. Clustering ────────────────────────────────────────────
    h(f"3. CLUSTERING  (threshold={threshold}, min_size={min_size})")
    faces = (
        db.query(UnknownFace)
        .filter(UnknownFace.status == "pending", UnknownFace.face_encoding.isnot(None))
        .order_by(UnknownFace.last_seen.desc())
        .limit(args.limit)
        .all()
    )

    face_data = []
    for f in faces:
        enc = deserialize_encoding(f.face_encoding)
        if enc is None or len(enc) == 0:
            continue
        norm = np.linalg.norm(enc)
        if norm > 0:
            face_data.append((f, enc / norm))

    print(f"  loaded                     : {len(faces)} (cap {args.limit})")
    print(f"  usable embeddings          : {len(face_data)}")

    groups = compute_cluster_groups(face_data, threshold)
    kept = {k: v for k, v in groups.items() if len(v) >= min_size}
    dropped = len(groups) - len(kept)
    clustered_faces = sum(len(v) for v in kept.values())

    print(f"  raw groups                 : {len(groups)}")
    print(f"  clusters >= min_size       : {len(kept)}")
    print(f"  groups below min_size      : {dropped}")
    print(f"  faces inside kept clusters : {clustered_faces}"
          f"   ({100.0 * clustered_faces / max(len(face_data), 1):.1f}% of usable)")

    h("4. CLUSTER SIZE DISTRIBUTION")
    histogram(Counter(bucket_size(len(v)) for v in groups.values()),
              total=len(groups))

    # ── 5. Outliers ──────────────────────────────────────────────
    h("5. OUTLIER DISTRIBUTION (kept clusters)")
    outlier_counts = Counter()
    total_outliers = 0
    for members in kept.values():
        centroid = np.mean([e for _, e in members], axis=0)
        dists = centroid_distances([e for _, e in members], centroid)
        flags, _ = outlier_flags(dists, settings.GALLERY_OPT_OUTLIER_FACTOR)
        n = sum(flags)
        total_outliers += n
        outlier_counts[f"{n} outlier(s)"] += 1
    histogram(outlier_counts, total=len(kept))
    print(f"\n  total flagged members      : {total_outliers}"
          f"   ({100.0 * total_outliers / max(clustered_faces, 1):.1f}% of clustered faces)")

    # ── 6. Similarity to known persons ───────────────────────────
    h("6. SIMILARITY TO NEAREST ENROLLED PERSON")
    encodings = db.query(FaceEncoding).filter(FaceEncoding.encoding_data.isnot(None)).all()
    person_names = dict(db.query(Person.id, Person.name).all())
    print(f"  enrolled persons           : {len(set(e.person_id for e in encodings))}")
    print(f"  gallery encodings          : {len(encodings)}")

    gallery = []
    for e in encodings:
        arr = deserialize_encoding(e.encoding_data)
        if arr is None:
            continue
        n = np.linalg.norm(arr)
        if n > 0:
            gallery.append((e.person_id, arr / n))

    if not gallery:
        print("\n  No gallery encodings — no candidates can be produced.")
        db.close()
        return 0

    gal_matrix = np.array([g[1] for g in gallery], dtype=np.float32)
    gal_pids = [g[0] for g in gallery]

    sim_bands = Counter()
    gate_counts = Counter()
    top_hits = []
    per_person_best = {}

    for label, members in kept.items():
        centroid = np.mean([e for _, e in members], axis=0)
        cn = np.linalg.norm(centroid)
        if cn == 0:
            continue
        sims = gal_matrix @ (centroid / cn)

        # Best similarity per person, then top-K, exactly like the worker.
        best_per_person = {}
        for pid, s in zip(gal_pids, sims):
            if pid not in best_per_person or s > best_per_person[pid]:
                best_per_person[pid] = float(s)
        ranked = sorted(best_per_person.items(), key=lambda kv: -kv[1])[
            : settings.IDENTITY_REVIEW_TOP_K
        ]

        if ranked:
            sim_bands[similarity_band(ranked[0][1])] += 1
            top_hits.append((ranked[0][1], label, ranked[0][0], len(members)))

        for pid, s in ranked:
            gate_counts[worker._confidence_gate(s)] += 1
            if s > per_person_best.get(pid, 0.0):
                per_person_best[pid] = s

    print("\n  Best-match similarity per cluster:")
    histogram(sim_bands, total=len(kept))

    # ── 7. What a real cycle would do ────────────────────────────
    h("7. PROJECTED WORKER CYCLE")
    n_reject = gate_counts.get("reject", 0)
    n_vlm = gate_counts.get("call_vlm", 0)
    n_skip = gate_counts.get("skip_vlm_queue", 0)
    n_auto = gate_counts.get("auto_queue", 0)
    total_pairs = sum(gate_counts.values())

    print(f"  (cluster, person) pairs    : {total_pairs}"
          f"   [top_k={settings.IDENTITY_REVIEW_TOP_K}]")
    print(f"  rejected below safety valve: {n_reject}")
    print(f"  high tier, no VLM needed   : {n_skip + n_auto}"
          f"   (skip_vlm={n_skip}, auto_queue={n_auto})")
    print(f"  ambiguous, WANT a VLM call : {n_vlm}")

    budget = settings.IDENTITY_REVIEW_VLM_MAX_PER_CYCLE
    per_cycle = min(budget, n_vlm)
    print(f"\n  VLM budget per cycle       : {budget}")
    print(f"  VLM calls in cycle #1      : {per_cycle}")
    if n_vlm > budget:
        print(f"  cycles to clear VLM queue  : ~{-(-n_vlm // max(budget, 1))}"
              f"   ({settings.IDENTITY_REVIEW_INTERVAL_SEC}s apart)")
    print(f"  VLM provider               : {settings.IDENTITY_REVIEW_VLM_PROVIDER}")

    fairness = settings.IDENTITY_REVIEW_MAX_CLUSTERS_PER_PERSON
    capped = max(0, len(per_person_best) * fairness)
    expected = min(n_skip + n_auto + per_cycle, capped) if per_person_best else 0
    print(f"\n  per-person fairness cap    : {fairness} cluster(s)/person/cycle")
    print(f"  distinct persons matched   : {len(per_person_best)}")
    print(f"  review candidates, cycle #1: ~{expected}"
          f"   (upper bound {n_skip + n_auto + per_cycle})")

    # ── 8. Strongest clusters ────────────────────────────────────
    h("8. STRONGEST CLUSTER MATCHES (top 15)")
    top_hits.sort(reverse=True)
    print(f"  {'sim':>6}  {'members':>7}  person")
    for sim, label, pid, n in top_hits[:15]:
        gate = worker._confidence_gate(sim)
        print(f"  {sim:6.3f}  {n:>7}  {person_names.get(pid, pid)[:34]:<34} [{gate}]")

    print(f"\n{BAR}\n  Nothing was written. No VLM calls were made.\n{BAR}")
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
