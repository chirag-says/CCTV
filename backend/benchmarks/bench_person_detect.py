"""
PART 1+2+3: Person Detection Benchmark
=======================================
Benchmarks person detection on real CCTV frames at multiple resolutions.

READ-ONLY: loads models independently, does not touch production config.

Usage:
    python -m benchmarks.bench_person_detect

Output:
    benchmarks/results/person_detection.json
"""

import sys
import os
import time
import json
from pathlib import Path
from collections import defaultdict

# Ensure backend is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch

from benchmarks.bench_utils import (
    FRAMES_DIR, RESULTS_DIR, Timer, GPUSampler, gpu_snapshot, cpu_snapshot,
    load_frames_from_dir, save_results, bbox_height, bucket_value,
    HEIGHT_BUCKETS,
)

# ── Configuration ─────────────────────────────────────────────────
MODELS_TO_TEST = ["yolov8n.pt", "yolo11m.pt"]
IMGSZ_TO_TEST = [640, 960, 1280]
MAX_FRAMES = 200
WARMUP_FRAMES = 10
CONF_THRESH = 0.35  # match production PERSON_DET_CONF


def run_person_detection_benchmark():
    """Run person detection across models × resolutions on real frames."""
    print("=" * 70)
    print("PART 2: PERSON DETECTION BENCHMARK")
    print("=" * 70)

    # Load test frames
    print(f"\nLoading frames from {FRAMES_DIR} ...")
    frame_data = load_frames_from_dir(FRAMES_DIR, max_frames=MAX_FRAMES, seed=42)
    print(f"  Loaded {len(frame_data)} frames")
    if not frame_data:
        print("ERROR: No frames found!")
        return

    # Document source
    resolutions = defaultdict(int)
    for fname, frame in frame_data:
        h, w = frame.shape[:2]
        resolutions[f"{w}x{h}"] += 1
    print(f"  Resolutions: {dict(resolutions)}")

    # Store all results
    all_results = {
        "test_metadata": {
            "n_frames": len(frame_data),
            "resolutions": dict(resolutions),
            "conf_threshold": CONF_THRESH,
            "warmup_frames": WARMUP_FRAMES,
            "gpu_before": gpu_snapshot(),
        },
        "experiments": [],
    }

    from ultralytics import YOLO

    for model_name in MODELS_TO_TEST:
        for imgsz in IMGSZ_TO_TEST:
            experiment_id = f"{model_name.replace('.pt', '')}_{imgsz}"
            print(f"\n{'─' * 60}")
            print(f"Testing: {model_name} @ {imgsz}px (fp16)")
            print(f"{'─' * 60}")

            # Load model fresh — isolated from production
            model = YOLO(model_name)

            # Warmup
            print(f"  Warming up ({WARMUP_FRAMES} frames) ...")
            for i in range(min(WARMUP_FRAMES, len(frame_data))):
                model.predict(
                    frame_data[i][1],
                    imgsz=imgsz,
                    half=True,
                    classes=[0],  # person only
                    conf=CONF_THRESH,
                    verbose=False,
                )

            gpu_before = gpu_snapshot()
            gpu_sampler = GPUSampler()
            gpu_sampler.start()

            timer = Timer("predict")
            preproc_timer = Timer("preproc")

            # Per-frame results
            frame_results = []
            height_bucket_counts = defaultdict(int)
            total_detections = 0

            for frame_idx, (fname, frame) in enumerate(frame_data):
                h_orig, w_orig = frame.shape[:2]

                with timer.measure():
                    results = model.predict(
                        frame,
                        imgsz=imgsz,
                        half=True,
                        classes=[0],
                        conf=CONF_THRESH,
                        verbose=False,
                    )

                # Extract detections
                boxes = results[0].boxes
                n_det = len(boxes)
                total_detections += n_det

                det_info = []
                for b in boxes:
                    xyxy = b.xyxy[0].cpu().numpy()
                    conf = float(b.conf[0])
                    h_px = bbox_height(xyxy)
                    bucket = bucket_value(h_px, HEIGHT_BUCKETS)
                    height_bucket_counts[bucket] += 1
                    det_info.append({
                        "bbox": [int(x) for x in xyxy],
                        "conf": round(conf, 3),
                        "height_px": h_px,
                        "bucket": bucket,
                    })

                frame_results.append({
                    "frame": fname[:40],
                    "resolution": f"{w_orig}x{h_orig}",
                    "n_detections": n_det,
                    "detections": det_info,
                })

            gpu_stats = gpu_sampler.stop()
            gpu_after = gpu_snapshot()
            cpu_stats = cpu_snapshot()

            experiment = {
                "id": experiment_id,
                "model": model_name,
                "imgsz": imgsz,
                "precision": "fp16",
                "total_frames": len(frame_data),
                "total_detections": total_detections,
                "detections_per_frame": round(total_detections / len(frame_data), 2),
                "timing_ms": timer.stats,
                "effective_fps": round(1000.0 / timer.stats["avg"], 1) if timer.stats["avg"] > 0 else 0,
                "height_bucket_counts": dict(height_bucket_counts),
                "gpu_stats": gpu_stats,
                "gpu_after": gpu_after,
                "cpu_stats": cpu_stats,
                "frame_results": frame_results,
            }

            print(f"  Detections: {total_detections} total, {experiment['detections_per_frame']}/frame")
            print(f"  Timing:     avg={timer.stats['avg']:.1f}ms  p95={timer.stats['p95']:.1f}ms  max={timer.stats['max']:.1f}ms")
            print(f"  FPS:        {experiment['effective_fps']}")
            print(f"  Buckets:    {dict(height_bucket_counts)}")
            print(f"  GPU:        util={gpu_stats.get('gpu_util_avg', '?')}%  vram={gpu_stats.get('vram_max_mb', '?')}MB")

            all_results["experiments"].append(experiment)

            # Cleanup
            del model
            torch.cuda.empty_cache()
            time.sleep(1)

    save_results("person_detection", all_results)
    print(f"\n✅ Person detection benchmark complete.")
    return all_results


if __name__ == "__main__":
    run_person_detection_benchmark()
