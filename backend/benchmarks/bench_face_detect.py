"""
PART 4+5: Face Detection & Recognition Gate Benchmark
=====================================================
Tests SCRFD face detection and measures per-size-bucket face detectability,
then tests recognition accuracy at different face-size gates.

READ-ONLY: loads insightface independently, does not touch production config.

Usage:
    python -m benchmarks.bench_face_detect

Output:
    benchmarks/results/face_detection.json
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
    FRAMES_DIR, Timer, GPUSampler, gpu_snapshot, cpu_snapshot,
    load_frames_from_dir, save_results, bucket_value,
    FACE_SIZE_BUCKETS,
)


# ── Configuration ─────────────────────────────────────────────────
MAX_FRAMES = 200
WARMUP_FRAMES = 5
DET_SIZES_TO_TEST = [640, 1024]
DET_THRESH = 0.5
# Recognition gate sizes to test (Part 5)
RECOGNITION_GATES = [50, 60, 70, 80, 90, 100, 120]


def _load_insightface(det_size: int):
    """Load insightface FaceAnalysis with specific det_size, isolated from production."""
    # Replicate the CUDA DLL fix from detector.py
    import ctypes
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
    app.prepare(ctx_id=0, det_size=(det_size, det_size), det_thresh=DET_THRESH)
    return app


def run_face_detection_benchmark():
    """Run face detection across det_size configurations on real frames."""
    print("=" * 70)
    print("PART 4: FACE DETECTION BENCHMARK")
    print("=" * 70)

    # Load test frames
    print(f"\nLoading frames from {FRAMES_DIR} ...")
    frame_data = load_frames_from_dir(FRAMES_DIR, max_frames=MAX_FRAMES, seed=42)
    print(f"  Loaded {len(frame_data)} frames")
    if not frame_data:
        print("ERROR: No frames found!")
        return

    all_results = {
        "test_metadata": {
            "n_frames": len(frame_data),
            "det_thresh": DET_THRESH,
            "recognition_gates_tested": RECOGNITION_GATES,
        },
        "experiments": [],
    }

    for det_size in DET_SIZES_TO_TEST:
        experiment_id = f"scrfd10g_{det_size}"
        print(f"\n{'─' * 60}")
        print(f"Testing: SCRFD-10G @ det_size={det_size}")
        print(f"{'─' * 60}")

        app = _load_insightface(det_size)

        # Warmup
        print(f"  Warming up ({WARMUP_FRAMES} frames) ...")
        for i in range(min(WARMUP_FRAMES, len(frame_data))):
            app.get(frame_data[i][1])

        gpu_sampler = GPUSampler()
        gpu_sampler.start()

        detect_timer = Timer("detect")
        embed_timer = Timer("embed")

        frame_results = []
        face_size_counts = defaultdict(int)
        total_faces = 0
        total_with_embedding = 0
        faces_by_gate = defaultdict(lambda: {"detected": 0, "has_embedding": 0})

        for fname, frame in frame_data:
            h_orig, w_orig = frame.shape[:2]

            with detect_timer.measure():
                faces = app.get(frame)

            n_faces = len(faces)
            total_faces += n_faces

            face_info = []
            for face in faces:
                bbox = face.bbox.astype(int)
                face_w = int(bbox[2] - bbox[0])
                face_h = int(bbox[3] - bbox[1])
                det_score = float(face.det_score)
                has_embedding = face.embedding is not None
                if has_embedding:
                    total_with_embedding += 1

                bucket = bucket_value(min(face_w, face_h), FACE_SIZE_BUCKETS)
                face_size_counts[bucket] += 1

                # Part 5: Test each recognition gate
                min_dim = min(face_w, face_h)
                for gate in RECOGNITION_GATES:
                    faces_by_gate[gate]["detected"] += 1
                    if min_dim >= gate and has_embedding:
                        faces_by_gate[gate]["has_embedding"] += 1

                face_info.append({
                    "bbox": [int(x) for x in bbox],
                    "face_w": face_w,
                    "face_h": face_h,
                    "min_dim": min_dim,
                    "det_score": round(det_score, 4),
                    "has_embedding": has_embedding,
                    "bucket": bucket,
                    "landmarks": face.kps.tolist() if face.kps is not None else None,
                })

            frame_results.append({
                "frame": fname[:40],
                "resolution": f"{w_orig}x{h_orig}",
                "n_faces": n_faces,
                "faces": face_info,
            })

        gpu_stats = gpu_sampler.stop()

        # Recognition gate analysis (Part 5)
        gate_analysis = {}
        for gate in RECOGNITION_GATES:
            g = faces_by_gate[gate]
            gate_analysis[str(gate)] = {
                "total_detected": g["detected"],
                "passes_gate": g["has_embedding"],
                "rejection_rate": round(
                    1.0 - g["has_embedding"] / g["detected"], 3
                ) if g["detected"] > 0 else 0,
            }

        experiment = {
            "id": experiment_id,
            "det_size": det_size,
            "total_frames": len(frame_data),
            "total_faces": total_faces,
            "faces_per_frame": round(total_faces / len(frame_data), 2),
            "faces_with_embedding": total_with_embedding,
            "timing_ms": detect_timer.stats,
            "effective_fps": round(1000.0 / detect_timer.stats["avg"], 1) if detect_timer.stats["avg"] > 0 else 0,
            "face_size_bucket_counts": dict(face_size_counts),
            "recognition_gate_analysis": gate_analysis,
            "gpu_stats": gpu_stats,
            "frame_results": frame_results,
        }

        print(f"  Faces:   {total_faces} total, {experiment['faces_per_frame']}/frame")
        print(f"  Timing:  avg={detect_timer.stats['avg']:.1f}ms  p95={detect_timer.stats['p95']:.1f}ms")
        print(f"  FPS:     {experiment['effective_fps']}")
        print(f"  Buckets: {dict(face_size_counts)}")
        print(f"  Recognition gate analysis:")
        for gate, info in gate_analysis.items():
            print(f"    {gate}px: {info['passes_gate']}/{info['total_detected']} pass ({info['rejection_rate']*100:.1f}% rejected)")
        print(f"  GPU:     util={gpu_stats.get('gpu_util_avg', '?')}%  vram={gpu_stats.get('vram_max_mb', '?')}MB")

        all_results["experiments"].append(experiment)

        # Cleanup
        del app
        torch.cuda.empty_cache()
        time.sleep(2)

    save_results("face_detection", all_results)
    print(f"\n✅ Face detection benchmark complete.")
    return all_results


if __name__ == "__main__":
    run_face_detection_benchmark()
