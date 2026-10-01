"""
Phase 1 Validation Benchmark — YOLOv8n vs YOLO11m Side-by-Side
==============================================================
Runs BOTH models on the EXACT SAME frames and compares:
- Which detections are shared (both models agree)
- Which are YOLO11m-only (new detections to validate)
- Which are YOLOv8n-only (regressions to investigate)
- Height bucket analysis of the differences
- Visual side-by-side samples for human inspection

READ-ONLY: uses no production code, loads models independently.

Usage:
    python -m benchmarks.bench_phase1_validate

Output:
    benchmarks/results/phase1_validation.json
    benchmarks/results/phase1_samples/  (annotated frame pairs)
"""

import sys
import os
import time
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch

from benchmarks.bench_utils import (
    FRAMES_DIR, VIDEOS_DIR, RESULTS_DIR, Timer, GPUSampler, gpu_snapshot,
    cpu_snapshot, load_frames_from_dir, load_frames_from_video, save_results,
    bbox_height, bucket_value, HEIGHT_BUCKETS,
)

# ── Configuration ─────────────────────────────────────────────────
MAX_FRAMES = 200
WARMUP_FRAMES = 10
CONF_THRESH = 0.35
IMGSZ = 640
IOU_MATCH_THRESH = 0.45   # IoU threshold to consider two boxes "the same person"
SAMPLE_FRAMES = 15        # number of annotated comparison frames to save
VIDEO_TRACK_FRAMES = 500  # frames for tracking comparison


def _iou(box_a, box_b):
    """Compute IoU between two xyxy boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def _match_detections(dets_a, dets_b, iou_thresh=IOU_MATCH_THRESH):
    """Match detections between two sets using greedy IoU matching.

    Returns:
        matched_pairs: list of (idx_a, idx_b, iou) tuples
        unmatched_a:   list of indices only in set A
        unmatched_b:   list of indices only in set B
    """
    if not dets_a or not dets_b:
        return [], list(range(len(dets_a))), list(range(len(dets_b)))

    # Compute IoU matrix
    n_a, n_b = len(dets_a), len(dets_b)
    iou_matrix = np.zeros((n_a, n_b))
    for i, da in enumerate(dets_a):
        for j, db in enumerate(dets_b):
            iou_matrix[i, j] = _iou(da["bbox"], db["bbox"])

    matched_pairs = []
    used_a = set()
    used_b = set()

    # Greedy matching: highest IoU first
    while True:
        if iou_matrix.size == 0:
            break
        max_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        max_iou = iou_matrix[max_idx]
        if max_iou < iou_thresh:
            break
        i, j = max_idx
        matched_pairs.append((int(i), int(j), float(max_iou)))
        used_a.add(i)
        used_b.add(j)
        iou_matrix[i, :] = 0
        iou_matrix[:, j] = 0

    unmatched_a = [i for i in range(n_a) if i not in used_a]
    unmatched_b = [j for j in range(n_b) if j not in used_b]
    return matched_pairs, unmatched_a, unmatched_b


def _draw_comparison(frame, dets_v8, dets_v11, matched, only_v8, only_v11, frame_label=""):
    """Draw annotated comparison frame showing matched/unique detections."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    # Matched detections: green boxes
    for i_a, i_b, iou in matched:
        box = dets_v11[i_b]["bbox"]
        cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (0, 200, 0), 2)
        ht = bbox_height(box)
        cv2.putText(vis, f"BOTH {ht}px", (box[0], box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

    # YOLOv8n-only: red boxes (regressions)
    for i in only_v8:
        box = dets_v8[i]["bbox"]
        cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        ht = bbox_height(box)
        cv2.putText(vis, f"v8 ONLY {ht}px", (box[0], box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # YOLO11m-only: cyan boxes (new detections)
    for j in only_v11:
        box = dets_v11[j]["bbox"]
        cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (255, 200, 0), 2)
        ht = bbox_height(box)
        cv2.putText(vis, f"11m NEW {ht}px", (box[0], box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

    # Legend
    cv2.putText(vis, f"{frame_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(vis, f"GREEN=both  CYAN=11m-new  RED=v8-only", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return vis


def run_phase1_validation():
    """Side-by-side YOLOv8n vs YOLO11m on identical frames."""
    print("=" * 70)
    print("PHASE 1 VALIDATION: YOLOv8n vs YOLO11m Side-by-Side")
    print("=" * 70)

    from ultralytics import YOLO

    # ── Load test frames ─────────────────────────────────────────
    print(f"\nLoading frames from {FRAMES_DIR} ...")
    frame_data = load_frames_from_dir(FRAMES_DIR, max_frames=MAX_FRAMES, seed=42)
    print(f"  Loaded {len(frame_data)} frames")

    # ── Create output directory for samples ──────────────────────
    samples_dir = RESULTS_DIR / "phase1_samples"
    samples_dir.mkdir(exist_ok=True)

    # ── Load both models ─────────────────────────────────────────
    print("\nLoading YOLOv8n ...")
    model_v8 = YOLO("yolov8n.pt")
    print("Loading YOLO11m ...")
    model_v11 = YOLO("yolo11m.pt")

    # Warmup both
    print(f"Warming up ({WARMUP_FRAMES} frames each) ...")
    for i in range(min(WARMUP_FRAMES, len(frame_data))):
        model_v8.predict(frame_data[i][1], imgsz=IMGSZ, half=True, classes=[0],
                         conf=CONF_THRESH, verbose=False)
        model_v11.predict(frame_data[i][1], imgsz=IMGSZ, half=True, classes=[0],
                          conf=CONF_THRESH, verbose=False)

    # ── Run both models on every frame ───────────────────────────
    print(f"\nRunning side-by-side on {len(frame_data)} frames ...")

    timer_v8 = Timer("v8")
    timer_v11 = Timer("v11")

    gpu_sampler_v8 = GPUSampler()
    gpu_sampler_v11 = GPUSampler()

    # Per-frame comparison
    frame_comparisons = []
    total_matched = 0
    total_v8_only = 0
    total_v11_only = 0
    total_v8 = 0
    total_v11 = 0

    # Height-bucket analysis for unique detections
    v11_new_buckets = defaultdict(int)
    v8_lost_buckets = defaultdict(int)
    matched_buckets = defaultdict(int)

    # Confidence analysis for v11-only detections
    v11_new_confs = []
    v11_new_heights = []

    # Track which frames have the most interesting differences
    interesting_frames = []

    # -- v8 pass --
    gpu_sampler_v8.start()
    all_v8_dets = []
    for fname, frame in frame_data:
        with timer_v8.measure():
            r = model_v8.predict(frame, imgsz=IMGSZ, half=True, classes=[0],
                                 conf=CONF_THRESH, verbose=False)
        dets = []
        for b in r[0].boxes:
            xyxy = b.xyxy[0].cpu().numpy().astype(int)
            dets.append({
                "bbox": [int(x) for x in xyxy],
                "conf": round(float(b.conf[0]), 4),
                "height": bbox_height(xyxy),
            })
        all_v8_dets.append(dets)
    gpu_v8 = gpu_sampler_v8.stop()

    # -- v11 pass --
    gpu_sampler_v11.start()
    all_v11_dets = []
    for fname, frame in frame_data:
        with timer_v11.measure():
            r = model_v11.predict(frame, imgsz=IMGSZ, half=True, classes=[0],
                                  conf=CONF_THRESH, verbose=False)
        dets = []
        for b in r[0].boxes:
            xyxy = b.xyxy[0].cpu().numpy().astype(int)
            dets.append({
                "bbox": [int(x) for x in xyxy],
                "conf": round(float(b.conf[0]), 4),
                "height": bbox_height(xyxy),
            })
        all_v11_dets.append(dets)
    gpu_v11 = gpu_sampler_v11.stop()

    # ── Compare frame by frame ───────────────────────────────────
    for idx, ((fname, frame), dets_v8, dets_v11) in enumerate(
            zip(frame_data, all_v8_dets, all_v11_dets)):

        matched, only_v8, only_v11 = _match_detections(dets_v8, dets_v11)

        n_matched = len(matched)
        n_v8_only = len(only_v8)
        n_v11_only = len(only_v11)
        total_matched += n_matched
        total_v8_only += n_v8_only
        total_v11_only += n_v11_only
        total_v8 += len(dets_v8)
        total_v11 += len(dets_v11)

        # Bucket analysis
        for i_a, i_b, iou in matched:
            h = dets_v11[i_b]["height"]
            matched_buckets[bucket_value(h, HEIGHT_BUCKETS)] += 1

        for i in only_v8:
            h = dets_v8[i]["height"]
            v8_lost_buckets[bucket_value(h, HEIGHT_BUCKETS)] += 1

        for j in only_v11:
            h = dets_v11[j]["height"]
            v11_new_buckets[bucket_value(h, HEIGHT_BUCKETS)] += 1
            v11_new_confs.append(dets_v11[j]["conf"])
            v11_new_heights.append(h)

        # Score for "interestingness" (more unique detections = more interesting)
        interest = n_v11_only + n_v8_only
        interesting_frames.append((interest, idx, fname, n_matched, n_v8_only, n_v11_only))

        frame_comparisons.append({
            "frame": fname[:50],
            "v8_count": len(dets_v8),
            "v11_count": len(dets_v11),
            "matched": n_matched,
            "v8_only": n_v8_only,
            "v11_only": n_v11_only,
        })

    # ── Save annotated sample frames ─────────────────────────────
    print(f"\nSaving {SAMPLE_FRAMES} annotated comparison frames ...")
    interesting_frames.sort(reverse=True)
    saved_samples = []

    for rank, (interest, idx, fname, n_m, n_v8o, n_v11o) in enumerate(
            interesting_frames[:SAMPLE_FRAMES]):
        if interest == 0:
            continue  # Skip frames with no differences
        frame = frame_data[idx][1]
        dets_v8 = all_v8_dets[idx]
        dets_v11 = all_v11_dets[idx]
        matched, only_v8, only_v11 = _match_detections(dets_v8, dets_v11)

        label = f"Frame {idx}: v8={len(dets_v8)} v11={len(dets_v11)} matched={n_m} v8only={n_v8o} 11mNEW={n_v11o}"
        vis = _draw_comparison(frame, dets_v8, dets_v11, matched, only_v8, only_v11, label)

        sample_path = samples_dir / f"compare_{rank:02d}_{fname[:30]}.jpg"
        cv2.imwrite(str(sample_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        saved_samples.append({
            "rank": rank,
            "frame": fname[:50],
            "path": str(sample_path),
            "v8": len(dets_v8),
            "v11": len(dets_v11),
            "matched": n_m,
            "v8_only": n_v8o,
            "v11_only": n_v11o,
            "v11_new_heights": [dets_v11[j]["height"] for j in only_v11],
            "v11_new_confs": [dets_v11[j]["conf"] for j in only_v11],
        })
        print(f"  [{rank}] {label}")

    # ── Classification of v11-only detections ────────────────────
    print(f"\n{'─' * 60}")
    print("DETECTION DIFFERENCE ANALYSIS")
    print(f"{'─' * 60}")

    # Duplicate detection check: look for v11-only boxes that overlap
    # with other v11 detections (would indicate duplicates)
    n_possible_duplicates = 0
    for idx in range(len(frame_data)):
        dets_v11 = all_v11_dets[idx]
        _, _, only_v11_indices = _match_detections(all_v8_dets[idx], dets_v11)
        for j in only_v11_indices:
            box_j = dets_v11[j]["bbox"]
            for k, det_k in enumerate(dets_v11):
                if k == j:
                    continue
                iou = _iou(box_j, det_k["bbox"])
                if iou > 0.30:  # >30% overlap with another v11 detection
                    n_possible_duplicates += 1
                    break

    # Small-person detection (<120px): these are "previously missed"
    small_new = sum(1 for h in v11_new_heights if h < 120)
    large_new = sum(1 for h in v11_new_heights if h >= 120)

    # Confidence distribution of new detections
    if v11_new_confs:
        low_conf_new = sum(1 for c in v11_new_confs if c < 0.5)
        high_conf_new = sum(1 for c in v11_new_confs if c >= 0.5)
    else:
        low_conf_new = high_conf_new = 0

    classification = {
        "total_v11_new_detections": total_v11_only,
        "possible_duplicates_iou30": n_possible_duplicates,
        "small_person_new_lt120px": small_new,
        "large_person_new_gte120px": large_new,
        "low_confidence_new_lt50": low_conf_new,
        "high_confidence_new_gte50": high_conf_new,
        "verdict": {
            "A_genuine_people": high_conf_new - n_possible_duplicates,
            "B_duplicate_detections": n_possible_duplicates,
            "C_false_positives_likely": low_conf_new,
            "D_previously_missed_small": small_new,
        },
    }

    print(f"  Total v11-only new detections: {total_v11_only}")
    print(f"  Possible duplicates (IoU>0.3): {n_possible_duplicates}")
    print(f"  Small persons (<120px):        {small_new}")
    print(f"  Large persons (>=120px):       {large_new}")
    print(f"  Low confidence (<0.5):         {low_conf_new}")
    print(f"  High confidence (>=0.5):       {high_conf_new}")
    print(f"\n  VERDICT:")
    for k, v in classification["verdict"].items():
        print(f"    {k}: {v}")

    # ── Tracking comparison ──────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("TRACKING COMPARISON (ByteTrack, same video)")
    print(f"{'─' * 60}")

    video_path = str(VIDEOS_DIR / "synth_people_720p.mp4")
    if not os.path.exists(video_path):
        video_path = str(VIDEOS_DIR / "d29c283e-f61e-4db2-a45d-f47da1994b62.mp4")

    tracking_comparison = {}

    if os.path.exists(video_path):
        from app.vision.bytetrack import ByteTracker

        for model_name, model in [("yolov8n", model_v8), ("yolo11m", model_v11)]:
            tracker = ByteTracker(track_thresh=0.5, track_buffer=90, match_thresh=0.8)
            cap = cv2.VideoCapture(video_path)

            track_ids = set()
            track_lifetimes = defaultdict(int)
            active_per_frame = []
            new_per_frame = []
            prev_ids = set()
            frame_idx = 0
            det_timer = Timer("det")
            track_timer = Timer("track")

            while frame_idx < VIDEO_TRACK_FRAMES:
                ret, frame = cap.read()
                if not ret:
                    break

                with det_timer.measure():
                    r = model.predict(frame, imgsz=IMGSZ, half=True, classes=[0],
                                      conf=CONF_THRESH, verbose=False)

                boxes = r[0].boxes
                if len(boxes) > 0:
                    bboxes = boxes.xyxy.cpu().numpy()
                    scores = boxes.conf.cpu().numpy()
                else:
                    bboxes = np.empty((0, 4))
                    scores = np.empty(0)

                with track_timer.measure():
                    active = tracker.update(bboxes, scores)

                curr_ids = set()
                for t in active:
                    tid = t.track_id
                    curr_ids.add(tid)
                    track_ids.add(tid)
                    track_lifetimes[tid] += 1

                new_ids = curr_ids - prev_ids
                active_per_frame.append(len(curr_ids))
                new_per_frame.append(len(new_ids))
                prev_ids = curr_ids
                frame_idx += 1

            cap.release()

            lifetimes = list(track_lifetimes.values())
            fragmented = sum(1 for l in lifetimes if l < 5)

            tracking_comparison[model_name] = {
                "frames_processed": frame_idx,
                "unique_tracks": len(track_ids),
                "total_track_creations": sum(new_per_frame),
                "fragmented_lt5": fragmented,
                "fragmentation_rate": round(fragmented / len(track_ids), 3) if track_ids else 0,
                "avg_lifetime": round(np.mean(lifetimes), 1) if lifetimes else 0,
                "median_lifetime": round(np.median(lifetimes), 1) if lifetimes else 0,
                "max_lifetime": int(max(lifetimes)) if lifetimes else 0,
                "active_per_frame_avg": round(np.mean(active_per_frame), 1),
                "detect_ms": det_timer.stats,
                "track_ms": track_timer.stats,
                "combined_fps": round(
                    1000.0 / (det_timer.stats["avg"] + track_timer.stats["avg"]), 1
                ) if (det_timer.stats["avg"] + track_timer.stats["avg"]) > 0 else 0,
            }

            print(f"\n  {model_name}:")
            tc = tracking_comparison[model_name]
            print(f"    Unique tracks:    {tc['unique_tracks']}")
            print(f"    Fragmented (<5):  {tc['fragmented_lt5']} ({tc['fragmentation_rate']*100:.1f}%)")
            print(f"    Avg lifetime:     {tc['avg_lifetime']}")
            print(f"    Detect avg:       {tc['detect_ms']['avg']:.1f}ms")
            print(f"    Combined FPS:     {tc['combined_fps']}")

    # ── Build final results ──────────────────────────────────────
    results = {
        "test_metadata": {
            "n_frames": len(frame_data),
            "conf_threshold": CONF_THRESH,
            "imgsz": IMGSZ,
            "iou_match_thresh": IOU_MATCH_THRESH,
            "gpu_before": gpu_snapshot(),
        },
        "detection_comparison": {
            "yolov8n": {
                "total_detections": total_v8,
                "per_frame": round(total_v8 / len(frame_data), 2),
                "timing_ms": timer_v8.stats,
                "effective_fps": round(1000.0 / timer_v8.stats["avg"], 1) if timer_v8.stats["avg"] > 0 else 0,
                "gpu_stats": gpu_v8,
                "cpu_stats": cpu_snapshot(),
            },
            "yolo11m": {
                "total_detections": total_v11,
                "per_frame": round(total_v11 / len(frame_data), 2),
                "timing_ms": timer_v11.stats,
                "effective_fps": round(1000.0 / timer_v11.stats["avg"], 1) if timer_v11.stats["avg"] > 0 else 0,
                "gpu_stats": gpu_v11,
                "cpu_stats": cpu_snapshot(),
            },
        },
        "matching_analysis": {
            "total_matched": total_matched,
            "total_v8_only": total_v8_only,
            "total_v11_only": total_v11_only,
            "matched_rate_v8": round(total_matched / total_v8, 3) if total_v8 > 0 else 0,
            "matched_rate_v11": round(total_matched / total_v11, 3) if total_v11 > 0 else 0,
            "v11_new_height_buckets": dict(v11_new_buckets),
            "v8_lost_height_buckets": dict(v8_lost_buckets),
            "matched_height_buckets": dict(matched_buckets),
        },
        "classification": classification,
        "v11_new_confidence_distribution": {
            "mean": round(float(np.mean(v11_new_confs)), 4) if v11_new_confs else None,
            "median": round(float(np.median(v11_new_confs)), 4) if v11_new_confs else None,
            "min": round(float(min(v11_new_confs)), 4) if v11_new_confs else None,
            "max": round(float(max(v11_new_confs)), 4) if v11_new_confs else None,
        },
        "v11_new_height_distribution": {
            "mean": round(float(np.mean(v11_new_heights)), 1) if v11_new_heights else None,
            "median": round(float(np.median(v11_new_heights)), 1) if v11_new_heights else None,
            "min": int(min(v11_new_heights)) if v11_new_heights else None,
            "max": int(max(v11_new_heights)) if v11_new_heights else None,
        },
        "tracking_comparison": tracking_comparison,
        "sample_frames_saved": saved_samples,
        "frame_comparisons": frame_comparisons,
    }

    save_results("phase1_validation", results)

    # ── Print summary table ──────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("PHASE 1 VALIDATION SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Metric':<35} {'YOLOv8n':<18} {'YOLO11m':<18} {'Delta':<15}")
    print("-" * 80)

    d_v8 = results["detection_comparison"]["yolov8n"]
    d_v11 = results["detection_comparison"]["yolo11m"]

    rows = [
        ("Detections/frame", d_v8["per_frame"], d_v11["per_frame"]),
        ("Total detections", d_v8["total_detections"], d_v11["total_detections"]),
        ("Avg inference (ms)", d_v8["timing_ms"]["avg"], d_v11["timing_ms"]["avg"]),
        ("P95 inference (ms)", d_v8["timing_ms"]["p95"], d_v11["timing_ms"]["p95"]),
        ("Effective FPS", d_v8["effective_fps"], d_v11["effective_fps"]),
    ]
    for label, v8_val, v11_val in rows:
        if isinstance(v8_val, float):
            delta = f"{v11_val - v8_val:+.2f}"
            print(f"{label:<35} {v8_val:<18.2f} {v11_val:<18.2f} {delta:<15}")
        else:
            delta = f"{v11_val - v8_val:+d}"
            print(f"{label:<35} {v8_val:<18} {v11_val:<18} {delta:<15}")

    print(f"\n  Matched detections (both agree): {total_matched}")
    print(f"  YOLOv8n-only (regressions):     {total_v8_only}")
    print(f"  YOLO11m-only (new):              {total_v11_only}")
    print(f"\n  v11-new by height bucket: {dict(v11_new_buckets)}")
    print(f"  v8-lost by height bucket: {dict(v8_lost_buckets)}")
    print(f"\n  Annotated samples saved to: {samples_dir}")
    print(f"{'=' * 80}")

    del model_v8, model_v11
    torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    run_phase1_validation()
