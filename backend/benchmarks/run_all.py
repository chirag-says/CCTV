"""
Master Benchmark Runner
========================
Runs all benchmark parts sequentially and produces the final comparison report.

READ-ONLY: No production code is modified. All results go to benchmarks/results/.

Usage:
    python -m benchmarks.run_all

Output:
    benchmarks/results/*.json           — individual benchmark results
    benchmarks/results/final_report.json — consolidated comparison table
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.bench_utils import RESULTS_DIR, gpu_snapshot, cpu_snapshot, save_results


def _header(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def run_all():
    """Execute all benchmarks and produce the final report."""
    start_time = time.time()
    timestamp = datetime.now().isoformat()

    _header("SentinelAI CCTV - COMPREHENSIVE BENCHMARK SUITE")
    print(f"Timestamp: {timestamp}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Initial GPU state: {gpu_snapshot()}")
    print(f"Initial CPU state: {cpu_snapshot()}")

    results = {}

    # ── Part 1+2+3: Person Detection ─────────────────────────────
    _header("PART 2+3: PERSON DETECTION")
    try:
        from benchmarks.bench_person_detect import run_person_detection_benchmark
        results["person_detection"] = run_person_detection_benchmark()
    except Exception as e:
        print(f"❌ Person detection benchmark failed: {e}")
        import traceback; traceback.print_exc()
        results["person_detection"] = {"error": str(e)}

    # ── Part 4+5: Face Detection ─────────────────────────────────
    _header("PART 4+5: FACE DETECTION")
    try:
        from benchmarks.bench_face_detect import run_face_detection_benchmark
        results["face_detection"] = run_face_detection_benchmark()
    except Exception as e:
        print(f"❌ Face detection benchmark failed: {e}")
        import traceback; traceback.print_exc()
        results["face_detection"] = {"error": str(e)}

    # ── Part 5+6: Face Recognition ───────────────────────────────
    _header("PART 5+6: FACE RECOGNITION & TEMPORAL")
    try:
        from benchmarks.bench_face_recog import run_face_recognition_benchmark
        results["face_recognition"] = run_face_recognition_benchmark()
    except Exception as e:
        print(f"❌ Face recognition benchmark failed: {e}")
        import traceback; traceback.print_exc()
        results["face_recognition"] = {"error": str(e)}

    # ── Part 7: Body ReID ────────────────────────────────────────
    _header("PART 7: BODY RE-ID")
    try:
        from benchmarks.bench_body_reid import run_body_reid_benchmark
        results["body_reid"] = run_body_reid_benchmark()
    except Exception as e:
        print(f"❌ Body ReID benchmark failed: {e}")
        import traceback; traceback.print_exc()
        results["body_reid"] = {"error": str(e)}

    # ── Part 8: Tracking ─────────────────────────────────────────
    _header("PART 8: TRACKING")
    try:
        from benchmarks.bench_tracking import run_tracking_benchmark
        results["tracking"] = run_tracking_benchmark()
    except Exception as e:
        print(f"❌ Tracking benchmark failed: {e}")
        import traceback; traceback.print_exc()
        results["tracking"] = {"error": str(e)}

    # ── Build Final Report ───────────────────────────────────────
    _header("FINAL REPORT")

    elapsed = time.time() - start_time
    print(f"\nTotal benchmark time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    final_report = {
        "timestamp": timestamp,
        "total_time_sec": round(elapsed, 1),
        "gpu_final": gpu_snapshot(),
        "cpu_final": cpu_snapshot(),
        "summary": _build_summary(results),
    }

    save_results("final_report", final_report)

    # Print summary table
    _print_summary_table(final_report["summary"])

    print(f"\n{'═' * 70}")
    print(f"  ALL BENCHMARKS COMPLETE")
    print(f"  Results: {RESULTS_DIR}")
    print(f"{'═' * 70}")

    return final_report


def _build_summary(results: dict) -> dict:
    """Build the consolidated comparison table from all benchmark results."""
    summary = {"components": []}

    # ── Person Detection Summary ─────────────────────────────────
    pd = results.get("person_detection", {})
    if "experiments" in pd:
        for exp in pd["experiments"]:
            summary["components"].append({
                "component": "Person Detector",
                "current_or_candidate": "current" if "yolov8n" in exp.get("model", "") and exp.get("imgsz") == 640 else "candidate",
                "configuration": f"{exp.get('model', '?')} @ {exp.get('imgsz', '?')}px fp16",
                "detections_per_frame": exp.get("detections_per_frame", 0),
                "latency_avg_ms": exp.get("timing_ms", {}).get("avg", 0),
                "latency_p95_ms": exp.get("timing_ms", {}).get("p95", 0),
                "effective_fps": exp.get("effective_fps", 0),
                "gpu_util_avg": exp.get("gpu_stats", {}).get("gpu_util_avg", 0),
                "vram_mb": exp.get("gpu_stats", {}).get("vram_max_mb", 0),
                "height_buckets": exp.get("height_bucket_counts", {}),
            })

    # ── Face Detection Summary ───────────────────────────────────
    fd = results.get("face_detection", {})
    if "experiments" in fd:
        for exp in fd["experiments"]:
            summary["components"].append({
                "component": "Face Detector",
                "current_or_candidate": "current" if exp.get("det_size") == 1024 else "candidate",
                "configuration": f"SCRFD-10G @ {exp.get('det_size', '?')}px",
                "faces_per_frame": exp.get("faces_per_frame", 0),
                "latency_avg_ms": exp.get("timing_ms", {}).get("avg", 0),
                "latency_p95_ms": exp.get("timing_ms", {}).get("p95", 0),
                "effective_fps": exp.get("effective_fps", 0),
                "gpu_util_avg": exp.get("gpu_stats", {}).get("gpu_util_avg", 0),
                "vram_mb": exp.get("gpu_stats", {}).get("vram_max_mb", 0),
                "face_size_buckets": exp.get("face_size_bucket_counts", {}),
            })

    # ── Face Recognition Summary ─────────────────────────────────
    fr = results.get("face_recognition", {})
    if "recognition_gate_analysis" in fr:
        summary["recognition_gate_analysis"] = fr["recognition_gate_analysis"]
    if "temporal_analysis" in fr:
        summary["temporal_analysis"] = fr["temporal_analysis"]

    # ── Body ReID Summary ────────────────────────────────────────
    br = results.get("body_reid", {})
    if "experiments" in br:
        for name, exp in br["experiments"].items():
            summary["components"].append({
                "component": "Body ReID",
                "current_or_candidate": "current" if "resnet50" in name else "candidate",
                "configuration": f"{name} ({exp.get('feat_dim', '?')}-dim)",
                "latency_avg_ms": exp.get("single_extract_ms", {}).get("avg", 0),
                "batch_per_crop_ms": exp.get("per_crop_batch_ms", 0),
                "gpu_util_avg": exp.get("gpu_stats", {}).get("gpu_util_avg", 0),
                "vram_mb": exp.get("gpu_stats", {}).get("vram_max_mb", 0),
                "sim_mean": exp.get("similarity_distribution", {}).get("mean", 0),
                "sim_std": exp.get("similarity_distribution", {}).get("std", 0),
            })

    # ── Tracking Summary ─────────────────────────────────────────
    tr = results.get("tracking", {})
    if "experiments" in tr:
        for exp in tr["experiments"]:
            cfg = exp.get("config", {})
            summary["components"].append({
                "component": "Tracker",
                "current_or_candidate": "current" if cfg.get("name") == "production_defaults" else "candidate",
                "configuration": f"ByteTrack ({cfg.get('name', '?')})",
                "unique_tracks": exp.get("total_unique_track_ids", 0),
                "fragmentation_rate": exp.get("fragmentation_rate", 0),
                "avg_lifetime": exp.get("track_lifetime_stats", {}).get("avg", 0),
                "tracking_ms": exp.get("tracking_timing_ms", {}).get("avg", 0),
                "combined_fps": exp.get("tracking_fps", 0),
            })

    return summary


def _print_summary_table(summary: dict):
    """Print the final comparison table."""
    print("\n" + "=" * 120)
    print(f"{'Component':<20} {'Config':<40} {'Type':<10} {'Latency(ms)':<12} {'FPS':<8} {'GPU%':<6} {'VRAM(MB)':<10} {'Key Metric'}")
    print("-" * 120)

    for c in summary.get("components", []):
        comp = c.get("component", "?")
        config = c.get("configuration", "?")[:38]
        ctype = c.get("current_or_candidate", "?")[:8]
        lat = c.get("latency_avg_ms", c.get("tracking_ms", 0))
        fps = c.get("effective_fps", c.get("combined_fps", 0))
        gpu = c.get("gpu_util_avg", 0)
        vram = c.get("vram_mb", 0)

        # Pick the most relevant metric
        key = ""
        if "detections_per_frame" in c:
            key = f"{c['detections_per_frame']} det/frame"
        elif "faces_per_frame" in c:
            key = f"{c['faces_per_frame']} face/frame"
        elif "fragmentation_rate" in c:
            key = f"frag={c['fragmentation_rate']*100:.1f}%"
        elif "sim_mean" in c:
            key = f"sim_mean={c['sim_mean']:.3f}"

        print(f"{comp:<20} {config:<40} {ctype:<10} {lat:<12.1f} {fps:<8} {gpu:<6} {vram:<10} {key}")

    print("=" * 120)

    # Recognition gate summary
    if "recognition_gate_analysis" in summary:
        print("\n  Recognition Gate Analysis:")
        for gate, info in summary["recognition_gate_analysis"].items():
            print(f"    {gate}px: {info.get('passes_gate', 0)} pass / "
                  f"{info.get('total_faces', 0)} total → "
                  f"match_rate={info.get('match_rate', 0)*100:.1f}%  "
                  f"gate_rejection={info.get('gate_rejection_rate', 0)*100:.1f}%")


if __name__ == "__main__":
    run_all()
