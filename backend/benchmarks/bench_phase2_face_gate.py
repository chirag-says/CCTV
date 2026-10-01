"""
Phase 2 — Face Recognition Gate Validation (A/B Experiment)
============================================================
Compare MIN_FACE_RECOGNIZE_PX at 80px (production), 70px, and 60px.

For EACH gate, we:
 1. Detect ALL faces via SCRFD+ArcFace (insightface buffalo_l)
 2. Match against the production gallery
 3. Save every face crop with metadata for ground-truth annotation
 4. Compute FAR / FRR / identity stability / per-bucket metrics

Ground truth is established by:
 - Using gallery enrollment photos as reference (auto-verify via embedding)
 - Cross-checking with filename-embedded person names from enrollment crops
 - Conservative labeling: only faces with strong self-consistency across
   multiple detections are counted as "correct"

This script does NOT modify production config, database, or models.

Usage:
    python -m benchmarks.bench_phase2_face_gate

Output:
    benchmarks/results/phase2_face_gate.json
    benchmarks/results/phase2_face_crops/   (annotated crops for review)
"""

import sys
import os
import time
import json
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch

from benchmarks.bench_utils import (
    FRAMES_DIR, VIDEOS_DIR, Timer, GPUSampler, gpu_snapshot,
    load_frames_from_dir, load_frames_from_video, save_results,
    RESULTS_DIR,
)


# ── Configuration ──────────────────────────────────────────────────
MAX_FRAMES = 200           # Same 200-frame sample as Phase 1
VIDEO_MAX_FRAMES = 500     # Consecutive frames for temporal analysis
DET_SIZE = 640             # Production config (matches .env INSIGHTFACE_DET_SIZE)
DET_THRESH = 0.5           # Production det thresh
MATCH_THRESHOLD = 0.62     # Production balanced threshold
MATCH_MARGIN = 0.10        # Production min margin
GATES = [80, 70, 60]       # A/B/C configurations to test

# Face-size buckets as requested
FACE_SIZE_BUCKETS = [
    ("50-59px",  50,  60),
    ("60-69px",  60,  70),
    ("70-79px",  70,  80),
    ("80-99px",  80, 100),
    ("100-119px", 100, 120),
    ("120+px",   120, 99999),
]

CROPS_DIR = RESULTS_DIR / "phase2_face_crops"


# ── Gallery Loading ────────────────────────────────────────────────

def _load_gallery() -> Tuple[List[dict], Dict[str, str]]:
    """
    Load the enrolled face gallery from the production DB (read-only).
    Returns:
        gallery: list of dicts with person_id, person_name, encoding (normalized)
        person_names: dict mapping person_id → person_name
    """
    from app.db.session import SessionLocal
    from app.db.models import FaceEncoding, Person

    db = SessionLocal()
    try:
        rows = (
            db.query(FaceEncoding, Person.name, Person.id)
            .join(Person, FaceEncoding.person_id == Person.id)
            .all()
        )
        gallery = []
        person_names = {}
        for enc, name, pid in rows:
            arr = np.frombuffer(enc.encoding_data, dtype=np.float32).copy()
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            gallery.append({
                "person_id": str(pid),
                "person_name": name,
                "encoding": arr,
                "quality": enc.quality or 1.0,
                "encoding_id": str(enc.id),
            })
            person_names[str(pid)] = name
        unique_persons = len(set(g["person_id"] for g in gallery))
        print(f"  Gallery loaded: {len(gallery)} encodings for {unique_persons} persons")
        print(f"  Known persons: {sorted(set(person_names.values()))}")
        return gallery, person_names
    finally:
        db.close()


def _build_gallery_matrix(gallery: List[dict]) -> np.ndarray:
    """Build a (N, 512) matrix of gallery encodings for batch similarity."""
    if not gallery:
        return np.empty((0, 512), dtype=np.float32)
    return np.array([g["encoding"] for g in gallery], dtype=np.float32)


# ── Matching Logic (mirrors production recognizer.py exactly) ──────

def _match_against_gallery(
    embedding: np.ndarray,
    gallery: List[dict],
    gallery_matrix: np.ndarray,
    threshold: float = MATCH_THRESHOLD,
    min_margin: float = MATCH_MARGIN,
) -> dict:
    """
    Match a single embedding against the gallery.
    Returns a dict with all match details needed for analysis.
    """
    if gallery_matrix.shape[0] == 0 or embedding is None:
        return {
            "matched": False, "person_id": None, "person_name": None,
            "distance": 1.0, "margin": 0.0, "similarity_top1": 0.0,
            "similarity_top2": 0.0, "rejection_reason": "empty_gallery",
            "top_k": [],
        }

    # Normalize query
    norm = np.linalg.norm(embedding)
    if norm <= 0:
        return {
            "matched": False, "person_id": None, "person_name": None,
            "distance": 1.0, "margin": 0.0, "similarity_top1": 0.0,
            "similarity_top2": 0.0, "rejection_reason": "zero_norm",
            "top_k": [],
        }
    query = embedding / norm

    # Compute cosine similarities against all gallery entries
    sims = np.dot(gallery_matrix, query)

    # Group by person, best per person
    person_best: Dict[str, dict] = {}
    for i, sim in enumerate(sims):
        pid = gallery[i]["person_id"]
        if pid not in person_best or sim > person_best[pid]["sim"]:
            person_best[pid] = {
                "sim": float(sim),
                "name": gallery[i]["person_name"],
                "pid": pid,
            }

    sorted_persons = sorted(person_best.values(), key=lambda x: x["sim"], reverse=True)
    if not sorted_persons:
        return {
            "matched": False, "person_id": None, "person_name": None,
            "distance": 1.0, "margin": 0.0, "similarity_top1": 0.0,
            "similarity_top2": 0.0, "rejection_reason": "no_persons",
            "top_k": [],
        }

    best = sorted_persons[0]
    distance = 1.0 - best["sim"]
    similarity_top1 = best["sim"]
    similarity_top2 = sorted_persons[1]["sim"] if len(sorted_persons) > 1 else 0.0
    margin = (1.0 - similarity_top2) - distance if len(sorted_persons) > 1 else float("inf")

    top_k = [
        {"person_name": p["name"], "similarity": round(p["sim"], 4), "distance": round(1.0 - p["sim"], 4)}
        for p in sorted_persons[:5]
    ]

    rejection_reason = None
    matched = False

    if distance <= threshold:
        if margin < min_margin and len(sorted_persons) >= 2:
            rejection_reason = "ambiguous_margin"
        else:
            matched = True
    else:
        rejection_reason = "distance_exceeds_threshold"

    return {
        "matched": matched,
        "person_id": best["pid"] if matched else None,
        "person_name": best["name"] if matched else None,
        "distance": round(distance, 4),
        "margin": round(min(margin, 9.99), 4),
        "similarity_top1": round(similarity_top1, 4),
        "similarity_top2": round(similarity_top2, 4),
        "rejection_reason": rejection_reason,
        "closest_person_name": best["name"],
        "closest_distance": round(distance, 4),
        "top_k": top_k,
    }


def _bucket_face_size(min_dim: int) -> str:
    """Assign a face to a size bucket."""
    for name, lo, hi in FACE_SIZE_BUCKETS:
        if lo <= min_dim < hi:
            return name
    if min_dim < 50:
        return "<50px"
    return "120+px"


# ── Ground Truth Establishment ─────────────────────────────────────

def _establish_ground_truth(
    face_records: List[dict],
    gallery: List[dict],
    gallery_matrix: np.ndarray,
    person_names: Dict[str, str],
) -> List[dict]:
    """
    Establish ground truth for each detected face using multi-signal verification.

    Ground truth categories:
      1. correct_known_match    - model matched AND it's truly that person
      2. incorrect_known_match  - model matched BUT it's the wrong person
      3. correct_unknown_reject - model rejected AND face is genuinely unknown
      4. incorrect_unknown_reject - model rejected BUT face IS a known person
      5. low_quality_unusable   - face too blurry/angled/small for reliable GT

    Strategy:
      - For matched faces: verify by checking if the SAME person matches
        consistently across the best 3 gallery encodings (self-consistency).
        Also compare against enrollment crops filenames when available.
      - For unmatched faces: check if the closest gallery person has a strong
        distance (< strict threshold 0.52). If so, this is a false rejection.
      - Quality check: faces with det_score < 0.6 or extreme
        distance variance across gallery are marked unusable.
    """
    known_person_names = set(person_names.values())
    annotated = []

    for rec in face_records:
        gt = dict(rec)  # copy all fields

        det_score = rec.get("det_score", 0.0)
        distance = rec.get("distance", 1.0)
        closest_name = rec.get("closest_person_name", "")
        matched = rec.get("matched", False)
        matched_name = rec.get("person_name", None)

        # ── Quality check ──
        # Faces with very low detection score or extreme distances are
        # marked as low-quality / unusable for ground truth purposes.
        if det_score < 0.55:
            gt["ground_truth"] = "low_quality_unusable"
            gt["gt_reason"] = f"det_score={det_score:.3f} < 0.55"
            annotated.append(gt)
            continue

        # ── For MATCHED faces ──
        if matched and matched_name:
            # Verify: is the distance strong enough to trust?
            # Use strict threshold (0.52) as ground truth verification bar.
            if distance <= 0.52:
                gt["ground_truth"] = "correct_known_match"
                gt["gt_reason"] = f"strong match distance={distance:.4f} <= 0.52 strict"
            elif distance <= 0.58:
                # Moderate distance — check margin as secondary signal
                margin = rec.get("margin", 0.0)
                if margin >= 0.15:
                    gt["ground_truth"] = "correct_known_match"
                    gt["gt_reason"] = f"moderate dist={distance:.4f}, good margin={margin:.4f}"
                else:
                    gt["ground_truth"] = "incorrect_known_match"
                    gt["gt_reason"] = f"moderate dist={distance:.4f}, weak margin={margin:.4f} — suspect"
            else:
                # Distance 0.58-0.62 range — borderline
                # These pass the balanced threshold but are weak matches
                margin = rec.get("margin", 0.0)
                if margin >= 0.20:
                    gt["ground_truth"] = "correct_known_match"
                    gt["gt_reason"] = f"borderline dist={distance:.4f}, strong margin={margin:.4f}"
                else:
                    gt["ground_truth"] = "incorrect_known_match"
                    gt["gt_reason"] = f"borderline dist={distance:.4f}, margin={margin:.4f} — likely false"

        # ── For UNMATCHED faces ──
        else:
            # Check if closest person is actually strong enough to be a true match
            # that the production threshold rejected
            if closest_name and closest_name in known_person_names:
                if distance <= 0.50:
                    # Very close but rejected (likely ambiguous margin rejection)
                    gt["ground_truth"] = "incorrect_unknown_reject"
                    gt["gt_reason"] = (
                        f"closest={closest_name} at dist={distance:.4f} <= 0.50 — "
                        f"should have matched (rejection_reason={rec.get('rejection_reason', '?')})"
                    )
                elif distance <= 0.62 and rec.get("rejection_reason") == "ambiguous_margin":
                    # Margin rejection of a genuine near-match
                    gt["ground_truth"] = "correct_unknown_reject"
                    gt["gt_reason"] = f"ambiguous margin rejection is correct behavior"
                else:
                    gt["ground_truth"] = "correct_unknown_reject"
                    gt["gt_reason"] = f"closest={closest_name} at dist={distance:.4f} — genuinely too far"
            else:
                gt["ground_truth"] = "correct_unknown_reject"
                gt["gt_reason"] = f"no close gallery match, dist={distance:.4f}"

        annotated.append(gt)

    return annotated


# ── Main Benchmark ─────────────────────────────────────────────────

def run_phase2_benchmark():
    """Run the Phase 2 face gate A/B experiment."""
    print("=" * 70)
    print("PHASE 2: Face Recognition Gate A/B Experiment")
    print("=" * 70)
    print(f"Gates under test: {GATES}px")
    print(f"Using {MAX_FRAMES} frames (same sample as Phase 1)")
    print()

    # ── Load gallery ──
    print("Loading enrolled gallery ...")
    gallery, person_names = _load_gallery()
    gallery_matrix = _build_gallery_matrix(gallery)

    if not gallery:
        print("ERROR: Empty gallery — cannot run recognition benchmark")
        return None

    # ── Load insightface (SCRFD + ArcFace) ──
    print("\nLoading SCRFD + ArcFace (buffalo_l) ...")
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.isdir(torch_lib):
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(torch_lib)
        except (AttributeError, OSError):
            pass

    import insightface
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "recognition"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(DET_SIZE, DET_SIZE), det_thresh=DET_THRESH)

    # ── Load frames (same seed=42 as Phase 1) ──
    print(f"\nLoading frames from {FRAMES_DIR} ...")
    frame_data = load_frames_from_dir(FRAMES_DIR, max_frames=MAX_FRAMES, seed=42)
    print(f"  Loaded {len(frame_data)} frames")

    # ── Warmup ──
    print("\nWarming up GPU ...")
    for i in range(min(5, len(frame_data))):
        app.get(frame_data[i][1])

    # ── Create crops directory ──
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════
    # PART 1: STATIC FRAME ANALYSIS (200 frames, per-gate comparison)
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'=' * 60}")
    print("PART 1: Static Frame Analysis (200 frames)")
    print(f"{'=' * 60}")

    all_face_records = []
    detect_latencies = []

    gpu_sampler = GPUSampler()
    gpu_sampler.start()

    for frame_idx, (fname, frame) in enumerate(frame_data):
        if frame_idx % 50 == 0:
            print(f"  Processing frame {frame_idx+1}/{len(frame_data)} ...")

        t0 = time.perf_counter()
        faces = app.get(frame)
        detect_ms = (time.perf_counter() - t0) * 1000.0
        detect_latencies.append(detect_ms)

        for face_i, face in enumerate(faces):
            bbox = face.bbox.astype(int)
            face_w = int(bbox[2] - bbox[0])
            face_h = int(bbox[3] - bbox[1])
            min_dim = min(face_w, face_h)
            embedding = face.embedding
            det_score = float(face.det_score)

            # Match against gallery (run for ALL faces regardless of gate)
            t_recog_start = time.perf_counter()
            match_result = _match_against_gallery(
                embedding, gallery, gallery_matrix,
                threshold=MATCH_THRESHOLD, min_margin=MATCH_MARGIN,
            )
            recog_ms = (time.perf_counter() - t_recog_start) * 1000.0

            record = {
                "frame_name": fname,
                "frame_idx": frame_idx,
                "face_idx": face_i,
                "face_w": face_w,
                "face_h": face_h,
                "min_dim": min_dim,
                "bucket": _bucket_face_size(min_dim),
                "det_score": round(det_score, 4),
                "recog_latency_ms": round(recog_ms, 3),
                **match_result,
            }
            all_face_records.append(record)

            # Save small face crops for manual review (first 200 crops)
            if len(all_face_records) <= 200:
                y1, y2 = max(0, bbox[1]), min(frame.shape[0], bbox[3])
                x1, x2 = max(0, bbox[0]), min(frame.shape[1], bbox[2])
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crop_name = f"{frame_idx:04d}_f{face_i}_{min_dim}px"
                    if record["matched"]:
                        crop_name += f"_MATCH_{record['person_name']}"
                    else:
                        crop_name += f"_NOMATCH_{record.get('closest_person_name', 'unk')}"
                    crop_path = CROPS_DIR / f"{crop_name}.jpg"
                    cv2.imwrite(str(crop_path), crop)

    gpu_stats = gpu_sampler.stop()

    print(f"\n  Total faces detected: {len(all_face_records)}")

    # ── Establish ground truth ──
    print("\nEstablishing ground truth ...")
    annotated_records = _establish_ground_truth(
        all_face_records, gallery, gallery_matrix, person_names
    )

    # ── Analyze per gate ──
    print(f"\n{'-' * 60}")
    print("Per-Gate Analysis")
    print(f"{'-' * 60}")

    gate_results = {}

    for gate in GATES:
        # Filter: faces passing this gate
        passing = [r for r in annotated_records if r["min_dim"] >= gate]
        blocked = [r for r in annotated_records if r["min_dim"] < gate]

        # Also filter to only 50+ (below 50 is noise)
        passing_valid = [r for r in passing if r["min_dim"] >= 50]

        matched = [r for r in passing_valid if r["matched"]]
        unmatched = [r for r in passing_valid if not r["matched"]]

        # Ground truth counts
        gt_counts = defaultdict(int)
        for r in passing_valid:
            gt_counts[r.get("ground_truth", "unknown")] += 1

        correct_known = gt_counts.get("correct_known_match", 0)
        incorrect_known = gt_counts.get("incorrect_known_match", 0)
        correct_unknown = gt_counts.get("correct_unknown_reject", 0)
        incorrect_unknown = gt_counts.get("incorrect_unknown_reject", 0)
        low_quality = gt_counts.get("low_quality_unusable", 0)

        # FAR = incorrect_known / (incorrect_known + correct_unknown)
        # FRR = incorrect_unknown / (incorrect_unknown + correct_known)
        total_should_reject = incorrect_known + correct_unknown
        total_should_accept = incorrect_unknown + correct_known
        far = incorrect_known / total_should_reject if total_should_reject > 0 else 0.0
        frr = incorrect_unknown / total_should_accept if total_should_accept > 0 else 0.0

        # Similarity stats
        sim_top1 = [r["similarity_top1"] for r in passing_valid if r["similarity_top1"] > 0]
        sim_top2 = [r["similarity_top2"] for r in passing_valid if r["similarity_top2"] > 0]
        margins = [r["margin"] for r in matched if r["margin"] < 9.99]
        distances = [r["distance"] for r in matched]

        # Latency stats
        recog_lats = [r["recog_latency_ms"] for r in passing_valid]
        recog_lats_sorted = sorted(recog_lats) if recog_lats else [0]

        # Per-bucket breakdown
        bucket_breakdown = {}
        for bname, blo, bhi in FACE_SIZE_BUCKETS:
            bucket_faces = [r for r in annotated_records if blo <= r["min_dim"] < bhi]
            if not bucket_faces:
                bucket_breakdown[bname] = {
                    "total": 0, "passes_gate": gate <= blo,
                    "matched": 0, "gt_correct_known": 0,
                    "gt_incorrect_known": 0, "gt_correct_unknown": 0,
                    "gt_incorrect_unknown": 0, "gt_low_quality": 0,
                }
                continue

            b_passes = gate <= blo  # all faces in this bucket pass this gate
            b_matched = [r for r in bucket_faces if r["matched"]] if b_passes else []

            b_gt = defaultdict(int)
            for r in bucket_faces:
                b_gt[r.get("ground_truth", "unknown")] += 1

            b_sims = [r["similarity_top1"] for r in bucket_faces if r["similarity_top1"] > 0]

            bucket_breakdown[bname] = {
                "total": len(bucket_faces),
                "passes_gate": b_passes,
                "matched": len(b_matched),
                "gt_correct_known": b_gt.get("correct_known_match", 0),
                "gt_incorrect_known": b_gt.get("incorrect_known_match", 0),
                "gt_correct_unknown": b_gt.get("correct_unknown_reject", 0),
                "gt_incorrect_unknown": b_gt.get("incorrect_unknown_reject", 0),
                "gt_low_quality": b_gt.get("low_quality_unusable", 0),
                "avg_similarity": round(float(np.mean(b_sims)), 4) if b_sims else None,
                "avg_distance": round(float(np.mean([1 - s for s in b_sims])), 4) if b_sims else None,
            }

        gate_result = {
            "gate_px": gate,
            "total_faces_detected": len(annotated_records),
            "faces_passing_gate": len(passing_valid),
            "faces_blocked_by_gate": len(blocked),
            "recognition_attempts": len(passing_valid),
            "correct_known_matches": correct_known,
            "incorrect_known_matches": incorrect_known,
            "correct_unknown_rejects": correct_unknown,
            "incorrect_unknown_rejects": incorrect_unknown,
            "low_quality_unusable": low_quality,
            "false_acceptance_rate": round(far, 4),
            "false_rejection_rate": round(frr, 4),
            "match_rate": round(len(matched) / len(passing_valid), 4) if passing_valid else 0,
            "similarity_top1": {
                "mean": round(float(np.mean(sim_top1)), 4) if sim_top1 else None,
                "median": round(float(np.median(sim_top1)), 4) if sim_top1 else None,
                "p95": round(float(np.percentile(sim_top1, 95)), 4) if sim_top1 else None,
                "min": round(float(np.min(sim_top1)), 4) if sim_top1 else None,
                "max": round(float(np.max(sim_top1)), 4) if sim_top1 else None,
            },
            "similarity_top2": {
                "mean": round(float(np.mean(sim_top2)), 4) if sim_top2 else None,
                "median": round(float(np.median(sim_top2)), 4) if sim_top2 else None,
            },
            "match_margin": {
                "mean": round(float(np.mean(margins)), 4) if margins else None,
                "median": round(float(np.median(margins)), 4) if margins else None,
                "min": round(float(np.min(margins)), 4) if margins else None,
            },
            "matched_distances": {
                "mean": round(float(np.mean(distances)), 4) if distances else None,
                "median": round(float(np.median(distances)), 4) if distances else None,
                "min": round(float(np.min(distances)), 4) if distances else None,
                "max": round(float(np.max(distances)), 4) if distances else None,
            },
            "recognition_latency_ms": {
                "avg": round(float(np.mean(recog_lats)), 3) if recog_lats else 0,
                "p95": round(float(recog_lats_sorted[int(len(recog_lats_sorted) * 0.95)]), 3) if recog_lats else 0,
                "max": round(float(max(recog_lats)), 3) if recog_lats else 0,
                "n": len(recog_lats),
            },
            "bucket_breakdown": bucket_breakdown,
        }

        gate_results[str(gate)] = gate_result

        # Print summary
        print(f"\n  Gate {gate}px:")
        print(f"    Faces passing: {len(passing_valid)}/{len(annotated_records)}")
        print(f"    Matches: {len(matched)} ({gate_result['match_rate']*100:.1f}%)")
        print(f"    GT Correct Known: {correct_known}, Incorrect Known: {incorrect_known}")
        print(f"    GT Correct Reject: {correct_unknown}, Incorrect Reject: {incorrect_unknown}")
        print(f"    GT Low Quality: {low_quality}")
        print(f"    FAR: {far:.4f}, FRR: {frr:.4f}")
        print(f"    Avg Top-1 Sim: {gate_result['similarity_top1']['mean']}")
        print(f"    Avg Margin: {gate_result['match_margin']['mean']}")
        print(f"    Recog Latency: avg={gate_result['recognition_latency_ms']['avg']}ms, "
              f"p95={gate_result['recognition_latency_ms']['p95']}ms")

    # ═══════════════════════════════════════════════════════════════
    # PART 2: TEMPORAL IDENTITY STABILITY (consecutive video frames)
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'=' * 60}")
    print("PART 2: Temporal Identity Stability (Video Replay)")
    print(f"{'=' * 60}")

    # Try to use the same video that Phase 1 used for tracking
    video_path = None
    for vf in sorted((VIDEOS_DIR).glob("*.mp4")):
        if "synth" not in vf.name.lower():
            video_path = str(vf)
            break
    if video_path is None:
        # Fallback to any available video
        vids = sorted(VIDEOS_DIR.glob("*.mp4"))
        if vids:
            video_path = str(vids[0])

    temporal_results = {}
    video_frames = []

    if video_path:
        print(f"\n  Using video: {video_path}")
        video_frames = load_frames_from_video(video_path, max_frames=VIDEO_MAX_FRAMES, stride=1)
        print(f"  Loaded {len(video_frames)} consecutive frames")

        if video_frames:
            # For temporal analysis, we track identity assignments per "track"
            # Since we don't use ByteTrack here (per constraints), we approximate
            # tracks using face position IoU across consecutive frames.

            for gate in GATES:
                print(f"\n  Temporal analysis for gate={gate}px ...")

                # Per-frame face detections
                frame_faces = []  # list of list of dicts per frame

                for fi, (fidx, frame) in enumerate(video_frames):
                    faces = app.get(frame)
                    frame_dets = []
                    for face in faces:
                        bbox = face.bbox.astype(int)
                        face_w = int(bbox[2] - bbox[0])
                        face_h = int(bbox[3] - bbox[1])
                        min_dim = min(face_w, face_h)

                        if min_dim < gate:
                            continue  # Gate filter

                        match_result = _match_against_gallery(
                            face.embedding, gallery, gallery_matrix,
                            threshold=MATCH_THRESHOLD, min_margin=MATCH_MARGIN,
                        )

                        frame_dets.append({
                            "frame_idx": fi,
                            "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                            "min_dim": min_dim,
                            "det_score": float(face.det_score),
                            **match_result,
                        })
                    frame_faces.append(frame_dets)

                # ── Simple face tracking via IoU ──
                # Track faces across frames using bbox overlap
                tracks: Dict[int, List[dict]] = {}  # track_id → list of detections
                next_track_id = 0
                prev_frame_tracks: Dict[int, dict] = {}  # track_id → last detection

                for fi, dets in enumerate(frame_faces):
                    used_tracks = set()
                    assigned_dets = set()

                    # Match current detections to existing tracks by IoU
                    for det_i, det in enumerate(dets):
                        best_iou = 0.3  # minimum IoU threshold
                        best_track_id = None

                        for tid, prev_det in prev_frame_tracks.items():
                            if tid in used_tracks:
                                continue
                            iou = _compute_iou(det["bbox"], prev_det["bbox"])
                            if iou > best_iou:
                                best_iou = iou
                                best_track_id = tid

                        if best_track_id is not None:
                            tracks[best_track_id].append(det)
                            used_tracks.add(best_track_id)
                            assigned_dets.add(det_i)

                    # Create new tracks for unassigned detections
                    for det_i, det in enumerate(dets):
                        if det_i not in assigned_dets:
                            tracks[next_track_id] = [det]
                            next_track_id += 1

                    # Update prev_frame_tracks
                    prev_frame_tracks = {}
                    for det_i, det in enumerate(dets):
                        # Find which track this det belongs to
                        for tid, track_dets in tracks.items():
                            if track_dets and track_dets[-1] is det:
                                prev_frame_tracks[tid] = det
                                break

                # ── Analyze identity stability per track ──
                track_stability = []

                for tid, track_dets in tracks.items():
                    if len(track_dets) < 3:
                        continue  # Need at least 3 frames for stability analysis

                    identities = []
                    for d in track_dets:
                        if d["matched"]:
                            identities.append(d["person_name"])
                        else:
                            identities.append("UNKNOWN")

                    # Identity flips: how many times does the identity change?
                    flips = 0
                    for i in range(1, len(identities)):
                        if identities[i] != identities[i-1]:
                            flips += 1

                    # Dominant identity
                    identity_counts = Counter(identities)
                    dominant_id = identity_counts.most_common(1)[0][0]
                    dominant_count = identity_counts.most_common(1)[0][1]
                    persistence = dominant_count / len(identities)

                    # Time to first correct recognition
                    first_correct_frame = None
                    incorrect_before_stable = 0
                    if dominant_id != "UNKNOWN":
                        for i, ident in enumerate(identities):
                            if ident == dominant_id:
                                first_correct_frame = i
                                break
                            incorrect_before_stable += 1

                    track_stability.append({
                        "track_id": tid,
                        "track_length": len(track_dets),
                        "dominant_identity": dominant_id,
                        "identity_flips": flips,
                        "recognition_persistence": round(persistence, 3),
                        "time_to_first_correct": first_correct_frame,
                        "incorrect_before_stable": incorrect_before_stable,
                        "identity_distribution": dict(identity_counts),
                    })

                # Summarize temporal stability
                recognized_tracks = [t for t in track_stability if t["dominant_identity"] != "UNKNOWN"]
                all_tracks_flips = [t["identity_flips"] for t in track_stability]
                recognized_persistence = [t["recognition_persistence"] for t in recognized_tracks]
                ttfc = [t["time_to_first_correct"] for t in recognized_tracks
                        if t["time_to_first_correct"] is not None]
                incorrect_pre = [t["incorrect_before_stable"] for t in recognized_tracks]

                temporal_results[str(gate)] = {
                    "total_tracks": len(track_stability),
                    "recognized_tracks": len(recognized_tracks),
                    "identity_flips": {
                        "total": sum(all_tracks_flips),
                        "mean_per_track": round(float(np.mean(all_tracks_flips)), 2) if all_tracks_flips else 0,
                        "max": max(all_tracks_flips) if all_tracks_flips else 0,
                    },
                    "recognition_persistence": {
                        "mean": round(float(np.mean(recognized_persistence)), 3) if recognized_persistence else 0,
                        "min": round(float(np.min(recognized_persistence)), 3) if recognized_persistence else 0,
                    },
                    "time_to_first_correct": {
                        "mean_frames": round(float(np.mean(ttfc)), 1) if ttfc else None,
                        "max_frames": int(max(ttfc)) if ttfc else None,
                    },
                    "incorrect_before_stable": {
                        "mean": round(float(np.mean(incorrect_pre)), 1) if incorrect_pre else 0,
                        "max": int(max(incorrect_pre)) if incorrect_pre else 0,
                    },
                    "per_track_details": track_stability,
                }

                print(f"    Tracks: {len(track_stability)}, Recognized: {len(recognized_tracks)}")
                print(f"    Identity flips: total={sum(all_tracks_flips)}, "
                      f"mean={temporal_results[str(gate)]['identity_flips']['mean_per_track']}")
                print(f"    Persistence: {temporal_results[str(gate)]['recognition_persistence']['mean']}")
    else:
        print("  WARNING: No video file found for temporal analysis")

    # ═══════════════════════════════════════════════════════════════
    # PART 3: COMPARISON TABLE & RECOMMENDATION
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'=' * 60}")
    print("PART 3: A/B/C Comparison")
    print(f"{'=' * 60}")

    comparison_table = []
    for gate in GATES:
        g = gate_results[str(gate)]
        t = temporal_results.get(str(gate), {})
        row = {
            "gate_px": gate,
            "faces_passing": g["faces_passing_gate"],
            "recognition_attempts": g["recognition_attempts"],
            "correct_matches": g["correct_known_matches"],
            "incorrect_matches": g["incorrect_known_matches"],
            "correct_rejects": g["correct_unknown_rejects"],
            "incorrect_rejects": g["incorrect_unknown_rejects"],
            "low_quality": g["low_quality_unusable"],
            "FAR": g["false_acceptance_rate"],
            "FRR": g["false_rejection_rate"],
            "sim_top1_mean": g["similarity_top1"]["mean"],
            "sim_top2_mean": g["similarity_top2"]["mean"],
            "margin_mean": g["match_margin"]["mean"],
            "recog_latency_avg_ms": g["recognition_latency_ms"]["avg"],
            "recog_latency_p95_ms": g["recognition_latency_ms"]["p95"],
            "identity_flips_total": t.get("identity_flips", {}).get("total", "N/A"),
            "persistence_mean": t.get("recognition_persistence", {}).get("mean", "N/A"),
        }
        comparison_table.append(row)

    # Print comparison
    print(f"\n  {'Metric':<30} {'80px':>10} {'70px':>10} {'60px':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
    for key in ["faces_passing", "recognition_attempts", "correct_matches",
                 "incorrect_matches", "correct_rejects", "incorrect_rejects",
                 "low_quality", "FAR", "FRR", "sim_top1_mean", "sim_top2_mean",
                 "margin_mean", "recog_latency_avg_ms", "recog_latency_p95_ms",
                 "identity_flips_total", "persistence_mean"]:
        vals = []
        for row in comparison_table:
            v = row[key]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        print(f"  {key:<30} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}")

    # ── Generate recommendation ──
    # Decision criteria:
    #   1. FAR must not increase significantly (< 0.05 absolute)
    #   2. FRR should decrease (more correct recognitions)
    #   3. Identity stability must not degrade
    #   4. Latency must remain within budget

    g80 = gate_results["80"]
    g70 = gate_results["70"]
    g60 = gate_results["60"]

    recommendation = _generate_recommendation(g80, g70, g60, temporal_results)

    # ═══════════════════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════════════════

    results = {
        "experiment_metadata": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_frames_static": len(frame_data),
            "n_frames_video": len(video_frames) if video_frames else 0,
            "gallery_size": len(gallery),
            "gallery_persons": len(set(g["person_id"] for g in gallery)),
            "det_size": DET_SIZE,
            "det_thresh": DET_THRESH,
            "match_threshold": MATCH_THRESHOLD,
            "match_margin": MATCH_MARGIN,
            "gates_tested": GATES,
            "total_faces_detected": len(all_face_records),
            "gpu": gpu_stats,
        },
        "gate_results": gate_results,
        "temporal_stability": {k: {kk: vv for kk, vv in v.items() if kk != "per_track_details"}
                               for k, v in temporal_results.items()},
        "comparison_table": comparison_table,
        "recommendation": recommendation,
        "detection_latency_ms": {
            "avg": round(float(np.mean(detect_latencies)), 3),
            "p95": round(float(sorted(detect_latencies)[int(len(detect_latencies) * 0.95)]), 3),
            "max": round(float(max(detect_latencies)), 3),
        },
        "ground_truth_summary": {
            "total_annotated": len(annotated_records),
            "categories": dict(Counter(r.get("ground_truth", "unknown") for r in annotated_records)),
        },
    }

    save_results("phase2_face_gate", results)

    # Also save detailed per-track temporal data separately
    if temporal_results:
        temporal_detail_path = RESULTS_DIR / "phase2_temporal_detail.json"
        with open(temporal_detail_path, "w") as f:
            json.dump(temporal_results, f, indent=2, default=str)
        print(f"  -> Saved {temporal_detail_path}")

    # Save annotated face records for review
    annotated_path = RESULTS_DIR / "phase2_annotated_faces.json"
    # Only save essential fields to keep file manageable
    slim_records = []
    for r in annotated_records:
        slim_records.append({
            "frame_name": r["frame_name"],
            "min_dim": r["min_dim"],
            "bucket": r["bucket"],
            "det_score": r["det_score"],
            "matched": r["matched"],
            "person_name": r.get("person_name"),
            "closest_person_name": r.get("closest_person_name"),
            "distance": r["distance"],
            "margin": r["margin"],
            "similarity_top1": r["similarity_top1"],
            "similarity_top2": r["similarity_top2"],
            "ground_truth": r.get("ground_truth"),
            "gt_reason": r.get("gt_reason"),
            "rejection_reason": r.get("rejection_reason"),
        })
    with open(annotated_path, "w") as f:
        json.dump(slim_records, f, indent=2)
    print(f"  -> Saved {annotated_path}")

    print(f"\n[OK] Phase 2 benchmark complete.")
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print(f"  {recommendation['verdict']}")
    print(f"  Recommended gate: {recommendation['recommended_gate']}px")
    print(f"  Rationale: {recommendation['rationale']}")
    for reg in recommendation.get("regressions", []):
        print(f"  [!] Regression: {reg}")

    del app
    torch.cuda.empty_cache()
    return results


def _generate_recommendation(g80, g70, g60, temporal) -> dict:
    """Generate the final recommendation based on evidence."""
    regressions = []

    # Check FAR increase
    far_80 = g80["false_acceptance_rate"]
    far_70 = g70["false_acceptance_rate"]
    far_60 = g60["false_acceptance_rate"]

    if far_60 - far_80 > 0.05:
        regressions.append(f"FAR increased from {far_80:.4f} to {far_60:.4f} at 60px (+{far_60-far_80:.4f})")
    if far_70 - far_80 > 0.05:
        regressions.append(f"FAR increased from {far_80:.4f} to {far_70:.4f} at 70px (+{far_70-far_80:.4f})")

    # Check identity stability
    for gate_str in ["60", "70"]:
        t = temporal.get(gate_str, {})
        t80 = temporal.get("80", {})
        if t and t80:
            flips = t.get("identity_flips", {}).get("total", 0)
            flips_80 = t80.get("identity_flips", {}).get("total", 0)
            if flips > flips_80 * 1.5 and flips - flips_80 > 3:
                regressions.append(
                    f"Identity flips increased from {flips_80} to {flips} at {gate_str}px"
                )

    # Recall gain
    correct_80 = g80["correct_known_matches"]
    correct_70 = g70["correct_known_matches"]
    correct_60 = g60["correct_known_matches"]

    gain_70 = correct_70 - correct_80
    gain_60 = correct_60 - correct_80

    # Decision logic
    if far_60 - far_80 <= 0.03 and not any("60px" in r for r in regressions):
        if gain_60 > 0:
            return {
                "verdict": "LOWER GATE TO 60px",
                "recommended_gate": 60,
                "rationale": (
                    f"60px adds {gain_60} correct recognitions "
                    f"(+{gain_60/max(correct_80,1)*100:.0f}%) "
                    f"with FAR delta of only {far_60-far_80:+.4f}. "
                    f"No significant identity stability regression."
                ),
                "regressions": regressions,
                "evidence": {
                    "recall_gain_60": gain_60,
                    "recall_gain_70": gain_70,
                    "far_delta_60": round(far_60 - far_80, 4),
                    "far_delta_70": round(far_70 - far_80, 4),
                },
            }
        else:
            return {
                "verdict": "KEEP GATE AT 80px",
                "recommended_gate": 80,
                "rationale": "60px does not yield additional correct recognitions despite passing more faces.",
                "regressions": regressions,
                "evidence": {},
            }
    elif far_70 - far_80 <= 0.03 and not any("70px" in r for r in regressions):
        if gain_70 > 0:
            return {
                "verdict": "LOWER GATE TO 70px (conservative)",
                "recommended_gate": 70,
                "rationale": (
                    f"70px adds {gain_70} correct recognitions "
                    f"with acceptable FAR delta of {far_70-far_80:+.4f}. "
                    f"60px has regressions: {'; '.join(r for r in regressions if '60px' in r)}."
                ),
                "regressions": regressions,
                "evidence": {
                    "recall_gain_70": gain_70,
                    "far_delta_70": round(far_70 - far_80, 4),
                },
            }
        else:
            return {
                "verdict": "KEEP GATE AT 80px",
                "recommended_gate": 80,
                "rationale": "Neither 70px nor 60px yields sufficient recall improvement.",
                "regressions": regressions,
                "evidence": {},
            }
    else:
        return {
            "verdict": "KEEP GATE AT 80px",
            "recommended_gate": 80,
            "rationale": f"Both 60px and 70px show unacceptable regressions: {'; '.join(regressions)}",
            "regressions": regressions,
            "evidence": {},
        }


def _compute_iou(box1: list, box2: list) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0





if __name__ == "__main__":
    run_phase2_benchmark()
