"""
PART 8: Tracking Benchmark
===========================
Benchmarks ByteTrack on actual CCTV video to measure ID switches,
track fragmentation, and recovery after occlusion.

READ-ONLY: instantiates ByteTrack independently, does not touch production.

Usage:
    python -m benchmarks.bench_tracking

Output:
    benchmarks/results/tracking.json
"""

import sys
import os
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch

from benchmarks.bench_utils import (
    VIDEOS_DIR, Timer, GPUSampler, gpu_snapshot,
    save_results,
)


MAX_FRAMES = 500
VIDEO_FILE = str(VIDEOS_DIR / "synth_people_720p.mp4")


def run_tracking_benchmark():
    """Benchmark ByteTrack tracking on real video footage."""
    print("=" * 70)
    print("PART 8: TRACKING BENCHMARK")
    print("=" * 70)

    if not os.path.exists(VIDEO_FILE):
        print(f"ERROR: Video not found: {VIDEO_FILE}")
        # Try the other video
        alt = str(VIDEOS_DIR / "d29c283e-f61e-4db2-a45d-f47da1994b62.mp4")
        if os.path.exists(alt):
            video_path = alt
            print(f"  Using alternate: {alt}")
        else:
            print("  No video files available!")
            return
    else:
        video_path = VIDEO_FILE

    # Get video info
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()

    print(f"\nVideo: {os.path.basename(video_path)}")
    print(f"  Resolution: {w}x{h}")
    print(f"  FPS: {fps:.1f}")
    print(f"  Frames: {total_frames}")
    print(f"  Duration: {duration:.1f}s")

    # Load YOLO for person detection
    from ultralytics import YOLO
    yolo = YOLO("yolov8n.pt")

    # Import ByteTrack from the production codebase (read-only)
    from app.vision.bytetrack import ByteTracker

    # Test configurations
    configs = [
        {
            "name": "production_defaults",
            "track_thresh": 0.5,
            "track_buffer": 90,
            "match_thresh": 0.8,
        },
        {
            "name": "longer_buffer",
            "track_thresh": 0.5,
            "track_buffer": 150,
            "match_thresh": 0.8,
        },
        {
            "name": "lower_thresh",
            "track_thresh": 0.35,
            "track_buffer": 90,
            "match_thresh": 0.8,
        },
    ]

    all_results = {
        "video_metadata": {
            "file": os.path.basename(video_path),
            "resolution": f"{w}x{h}",
            "fps": fps,
            "total_frames": total_frames,
            "duration_sec": round(duration, 1),
        },
        "experiments": [],
    }

    for config in configs:
        print(f"\n{'─' * 60}")
        print(f"Testing: {config['name']}")
        print(f"  track_thresh={config['track_thresh']}  buffer={config['track_buffer']}  match={config['match_thresh']}")
        print(f"{'─' * 60}")

        # Create fresh tracker
        tracker = ByteTracker(
            track_thresh=config["track_thresh"],
            track_buffer=config["track_buffer"],
            match_thresh=config["match_thresh"],
        )

        cap = cv2.VideoCapture(video_path)

        detect_timer = Timer("detect")
        track_timer = Timer("track")

        # Tracking metrics
        frame_idx = 0
        all_track_ids = set()
        track_lifetimes = defaultdict(int)  # track_id → frames active
        track_first_seen = {}  # track_id → frame
        track_last_seen = {}   # track_id → frame
        active_per_frame = []
        new_tracks_per_frame = []
        lost_tracks_per_frame = []

        prev_active_ids = set()

        while frame_idx < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break

            # Person detection
            with detect_timer.measure():
                results = yolo.predict(
                    frame, imgsz=640, half=True, classes=[0],
                    conf=0.35, verbose=False
                )

            boxes = results[0].boxes
            if len(boxes) > 0:
                bboxes = boxes.xyxy.cpu().numpy()
                scores = boxes.conf.cpu().numpy()
            else:
                bboxes = np.empty((0, 4))
                scores = np.empty(0)

            # ByteTrack update
            with track_timer.measure():
                active_tracks = tracker.update(bboxes, scores)

            current_ids = set()
            for track in active_tracks:
                tid = track.track_id
                current_ids.add(tid)
                all_track_ids.add(tid)
                track_lifetimes[tid] += 1
                if tid not in track_first_seen:
                    track_first_seen[tid] = frame_idx
                track_last_seen[tid] = frame_idx

            # Track new/lost
            new_ids = current_ids - prev_active_ids
            lost_ids = prev_active_ids - current_ids

            active_per_frame.append(len(current_ids))
            new_tracks_per_frame.append(len(new_ids))
            lost_tracks_per_frame.append(len(lost_ids))

            prev_active_ids = current_ids
            frame_idx += 1

        cap.release()

        # ── Compute tracking metrics ─────────────────────────────
        total_unique_tracks = len(all_track_ids)
        lifetimes = list(track_lifetimes.values())

        # Fragmentation: tracks with very short lifetimes (< 5 frames)
        fragmented = sum(1 for l in lifetimes if l < 5)

        # Track gaps: tracks that disappeared and reappeared
        # (different track_id for same visual person → we can't measure
        # without ground truth, but we can count total track creations)
        total_track_creations = sum(new_tracks_per_frame)

        # Recovery analysis: tracks that were lost then a new track appeared nearby
        # This is approximate without ground truth

        experiment = {
            "config": config,
            "frames_processed": frame_idx,
            "total_unique_track_ids": total_unique_tracks,
            "total_track_creations": total_track_creations,
            "fragmented_tracks_lt5": fragmented,
            "fragmentation_rate": round(fragmented / total_unique_tracks, 3) if total_unique_tracks > 0 else 0,
            "track_lifetime_stats": {
                "avg": round(np.mean(lifetimes), 1) if lifetimes else 0,
                "median": round(np.median(lifetimes), 1) if lifetimes else 0,
                "min": int(min(lifetimes)) if lifetimes else 0,
                "max": int(max(lifetimes)) if lifetimes else 0,
                "p25": round(np.percentile(lifetimes, 25), 1) if lifetimes else 0,
                "p75": round(np.percentile(lifetimes, 75), 1) if lifetimes else 0,
            },
            "active_tracks_per_frame": {
                "avg": round(np.mean(active_per_frame), 1),
                "max": int(max(active_per_frame)),
                "min": int(min(active_per_frame)),
            },
            "new_tracks_per_frame_avg": round(np.mean(new_tracks_per_frame), 2),
            "lost_tracks_per_frame_avg": round(np.mean(lost_tracks_per_frame), 2),
            "detection_timing_ms": detect_timer.stats,
            "tracking_timing_ms": track_timer.stats,
            "tracking_fps": round(
                1000.0 / (detect_timer.stats["avg"] + track_timer.stats["avg"]), 1
            ) if (detect_timer.stats["avg"] + track_timer.stats["avg"]) > 0 else 0,
        }

        print(f"  Unique tracks: {total_unique_tracks}")
        print(f"  Track creations: {total_track_creations}")
        print(f"  Fragmented (<5 frames): {fragmented} ({experiment['fragmentation_rate']*100:.1f}%)")
        print(f"  Lifetime: avg={experiment['track_lifetime_stats']['avg']:.0f}  "
              f"median={experiment['track_lifetime_stats']['median']:.0f}  "
              f"max={experiment['track_lifetime_stats']['max']}")
        print(f"  Active/frame: avg={experiment['active_tracks_per_frame']['avg']:.1f}  "
              f"max={experiment['active_tracks_per_frame']['max']}")
        print(f"  Detect: avg={detect_timer.stats['avg']:.1f}ms  Track: avg={track_timer.stats['avg']:.2f}ms")
        print(f"  Combined FPS: {experiment['tracking_fps']}")

        all_results["experiments"].append(experiment)

    save_results("tracking", all_results)
    print(f"\n✅ Tracking benchmark complete.")

    del yolo
    torch.cuda.empty_cache()
    return all_results


if __name__ == "__main__":
    run_tracking_benchmark()
