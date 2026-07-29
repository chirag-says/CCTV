"""
Performance Profiler — Structured aggregate performance monitoring.

Replaces noisy per-frame logging with periodic aggregate reports
that surface bottlenecks at a glance.

Reports every N total frames (default 100):
  - Per-stage timing with correct percentages (sum/sum) + P95
  - AI vs Non-AI time breakdown
  - Recognition cache effectiveness with reason breakdown
  - Event queue health
  - Snapshot / DB / WebSocket statistics
  - GPU utilization & memory
  - CPU usage
  - Camera FPS breakdown
  - Face detection statistics
  - JPEG compression ratio
  - Slow operation warnings (immediate)

Usage:
    from app.core.profiler import profiler

    with profiler.measure("scrfd_inference"):
        detect_faces()

    profiler.record_cache_reason("no_identity")
    profiler.tick_frame()
"""

import logging
import os
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger("cctv.profiler")


# ── Slow Operation Thresholds (ms) ────────────────────────────────
SLOW_THRESHOLDS = {
    "scrfd_inference": 25.0,
    "arcface": 15.0,
    "jpeg_encode": 20.0,
    "ws_send": 20.0,
    "frame_total": 50.0,
}


class PerformanceProfiler:
    """
    Collects per-stage timing and subsystem statistics.
    Prints aggregate reports every N total frames.
    Thread-safe.
    """

    def __init__(self, report_interval: int = 100, log_file: Optional[str] = None):
        self._enabled = True
        self._report_interval = report_interval
        self._lock = threading.Lock()
        self._log_file = log_file

        # Ensure log directory exists
        if self._log_file:
            os.makedirs(os.path.dirname(self._log_file) or ".", exist_ok=True)

        self._reset()

    def _reset(self):
        """Reset all counters for a new reporting interval (called under lock)."""
        # Per-stage timing: stage_name → list of durations (ms)
        self._timings: Dict[str, List[float]] = defaultdict(list)

        # Frame counters
        self._total_frames = 0
        self._detection_frames = 0
        self._report_start_time = time.perf_counter()

        # Camera stats
        self._frames_received = 0
        self._frames_skipped = 0
        self._frames_dropped = 0

        # Face detection stats
        self._faces_detected = 0

        # Recognition cache
        self._tracks_seen = 0
        self._recognition_calls = 0
        self._cache_hits = 0
        self._tracks_recreated = 0
        self._forced_rechecks = 0
        self._cache_reasons: Dict[str, int] = defaultdict(int)

        # Identity carryover stats
        self._carryover_success = 0
        self._carryover_failed = 0

        # ── Identity continuity (Phase 1) ─────────────────────────
        # Measures how often ByteTrack recreates a track, and how often a
        # recoverable predecessor existed at that moment. The gap between
        # _tracks_created and _tracks_recoverable is the headroom an identity
        # recovery layer can actually reclaim.
        self._tracks_created = 0
        self._tracks_recoverable = 0        # a plausible predecessor existed
        self._recovery_best_iou: List[float] = []
        self._recovery_best_dist: List[float] = []
        self._recovery_gap_frames: List[int] = []
        self._identity_swaps = 0            # person A -> person B on one track
        self._swap_examples: List[str] = []

        # Event queue
        self._eq_wait_times: List[float] = []
        self._eq_processed = 0
        self._eq_dropped = 0

        # Snapshots
        self._snapshots_saved = 0
        self._snapshot_times: List[float] = []
        self._snapshot_queue_len = 0

        # Database
        self._db_inserts = 0
        self._db_insert_times: List[float] = []
        self._db_failures = 0

        # WebSocket
        self._ws_frame_sizes: List[int] = []
        self._ws_send_times: List[float] = []
        self._ws_dropped = 0

        # JPEG compression stats
        self._raw_frame_sizes: List[int] = []

        # Slow operation warnings emitted this interval
        self._slow_warnings = 0

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ── Timing ────────────────────────────────────────────────────

    @contextmanager
    def measure(self, stage_name: str):
        """
        Context manager to measure a pipeline stage.

        Usage:
            with profiler.measure("scrfd_inference"):
                do_detection()
        """
        if not self._enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            with self._lock:
                self._timings[stage_name].append(elapsed_ms)

            # Slow operation detection — immediate warning
            threshold = SLOW_THRESHOLDS.get(stage_name)
            if threshold and elapsed_ms > threshold:
                self._emit_slow_warning(stage_name, elapsed_ms)

    def record_timing(self, stage_name: str, elapsed_ms: float):
        """Record a timing measurement directly (no context manager)."""
        if not self._enabled:
            return
        with self._lock:
            self._timings[stage_name].append(elapsed_ms)

        threshold = SLOW_THRESHOLDS.get(stage_name)
        if threshold and elapsed_ms > threshold:
            self._emit_slow_warning(stage_name, elapsed_ms)

    # ── Camera Stats ──────────────────────────────────────────────

    def record_frame_received(self):
        if not self._enabled:
            return
        with self._lock:
            self._frames_received += 1

    def record_frame_skipped(self):
        if not self._enabled:
            return
        with self._lock:
            self._frames_skipped += 1

    def record_frame_dropped(self):
        if not self._enabled:
            return
        with self._lock:
            self._frames_dropped += 1

    def record_detection_frame(self):
        """Called when a frame actually goes through detection (not skipped)."""
        if not self._enabled:
            return
        with self._lock:
            self._detection_frames += 1

    # ── Face Detection Stats ──────────────────────────────────────

    def record_faces_detected(self, count: int):
        """Called after SCRFD detection with the number of faces found."""
        if not self._enabled:
            return
        with self._lock:
            self._faces_detected += count

    # ── Recognition Cache ─────────────────────────────────────────

    def record_track_seen(self):
        if not self._enabled:
            return
        with self._lock:
            self._tracks_seen += 1

    def record_recognition_call(self):
        if not self._enabled:
            return
        with self._lock:
            self._recognition_calls += 1

    def record_cache_hit(self):
        if not self._enabled:
            return
        with self._lock:
            self._cache_hits += 1

    def record_track_recreated(self):
        if not self._enabled:
            return
        with self._lock:
            self._tracks_recreated += 1

    def record_forced_recheck(self):
        if not self._enabled:
            return
        with self._lock:
            self._forced_rechecks += 1

    def record_cache_reason(self, reason: str):
        """
        Record why recognition was triggered on a track.

        Reasons:
            no_identity     — track has no person_id yet
            new_track       — tracklet_len < 3 (just created/re-created)
            recheck_unlocked — time-based recheck on unlocked (low confidence) track
            recheck_locked  — forced recheck on locked (high confidence) track
        """
        if not self._enabled:
            return
        with self._lock:
            self._cache_reasons[reason] += 1

    def record_carryover_success(self):
        """Track identity was carried over from a lost/removed track."""
        if not self._enabled:
            return
        with self._lock:
            self._carryover_success += 1

    def record_carryover_failed(self):
        """New track was created without identity carryover."""
        if not self._enabled:
            return
        with self._lock:
            self._carryover_failed += 1

    # ── Identity Continuity (Phase 1) ─────────────────────────────

    def record_track_created(self, recoverable: bool, best_iou: float = 0.0,
                             best_dist: float = 0.0, gap_frames: int = 0):
        """
        Called once per newly created ByteTrack track.

        Args:
            recoverable: an eligible retired identity existed for this track
            best_iou: IoU of the best candidate (0.0 if none)
            best_dist: centroid distance in px of the best candidate
            gap_frames: detection-frames between retirement and recreation
        """
        if not self._enabled:
            return
        with self._lock:
            self._tracks_created += 1
            if recoverable:
                self._tracks_recoverable += 1
                self._recovery_best_iou.append(best_iou)
                self._recovery_best_dist.append(best_dist)
                self._recovery_gap_frames.append(gap_frames)

    def record_identity_swap(self, prev_name: str, new_name: str, track_id: int = -1):
        """
        Called when a single track's identity changes directly from one known
        person to a different known person, without passing through unknown.

        This is the metric that must NOT regress when identity recovery is
        enabled: in an attendance record a swap is strictly worse than a gap.
        """
        if not self._enabled:
            return
        with self._lock:
            self._identity_swaps += 1
            if len(self._swap_examples) < 5:
                self._swap_examples.append(f"T{track_id}: {prev_name} -> {new_name}")

    # ── Event Queue ───────────────────────────────────────────────

    def record_event_dispatched(self, wait_time_ms: float):
        if not self._enabled:
            return
        with self._lock:
            self._eq_wait_times.append(wait_time_ms)
            self._eq_processed += 1

    def record_event_dropped(self):
        if not self._enabled:
            return
        with self._lock:
            self._eq_dropped += 1

    # ── Snapshots ─────────────────────────────────────────────────

    def record_snapshot_save(self, duration_ms: float):
        if not self._enabled:
            return
        with self._lock:
            self._snapshots_saved += 1
            self._snapshot_times.append(duration_ms)

    def set_snapshot_queue_len(self, length: int):
        with self._lock:
            self._snapshot_queue_len = length

    # ── Database ──────────────────────────────────────────────────

    def record_db_insert(self, duration_ms: float):
        if not self._enabled:
            return
        with self._lock:
            self._db_inserts += 1
            self._db_insert_times.append(duration_ms)

    def record_db_failure(self):
        if not self._enabled:
            return
        with self._lock:
            self._db_failures += 1

    # ── WebSocket / JPEG ──────────────────────────────────────────

    def record_ws_send(self, frame_size_bytes: int, duration_ms: float):
        if not self._enabled:
            return
        with self._lock:
            self._ws_frame_sizes.append(frame_size_bytes)
            self._ws_send_times.append(duration_ms)

    def record_ws_drop(self):
        if not self._enabled:
            return
        with self._lock:
            self._ws_dropped += 1

    def record_raw_frame_size(self, size_bytes: int):
        """Record raw (uncompressed) frame size for compression ratio calculation."""
        if not self._enabled:
            return
        with self._lock:
            self._raw_frame_sizes.append(size_bytes)

    # ── Frame Tick ────────────────────────────────────────────────

    def tick_frame(self):
        """
        Called once per total frame (including skipped frames).
        Triggers aggregate report every N frames.
        """
        if not self._enabled:
            return

        with self._lock:
            self._total_frames += 1
            if self._total_frames >= self._report_interval:
                self._print_report()
                self._reset()

    # ── Slow Operation Warning ────────────────────────────────────

    def _emit_slow_warning(self, stage: str, duration_ms: float):
        """Emit an immediate warning for slow operations."""
        with self._lock:
            self._slow_warnings += 1
            frame_num = self._total_frames

        stage_labels = {
            "scrfd_inference": "SCRFD Inference",
            "arcface": "ArcFace Embed",
            "jpeg_encode": "JPEG Encode",
            "ws_send": "WebSocket Send",
            "frame_total": "Frame Total",
        }
        label = stage_labels.get(stage, stage)
        logger.warning(
            f"⚠ SLOW | Frame {frame_num} | {label}: {duration_ms:.1f}ms "
            f"(threshold: {SLOW_THRESHOLDS[stage]:.0f}ms)"
        )

    # ── GPU Stats ─────────────────────────────────────────────────

    @staticmethod
    def _get_gpu_stats() -> Optional[dict]:
        """Get GPU utilization stats. Tries pynvml first, falls back to torch."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            stats = {
                "utilization_pct": util.gpu,
                "mem_used_gb": mem_info.used / (1024 ** 3),
                "mem_total_gb": mem_info.total / (1024 ** 3),
            }

            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                stats["temperature_c"] = temp
            except Exception:
                pass

            pynvml.nvmlShutdown()
            return stats
        except Exception:
            pass

        # Fallback: torch.cuda (memory only, no utilization %)
        try:
            import torch
            if torch.cuda.is_available():
                mem_used = torch.cuda.memory_allocated(0) / (1024 ** 3)
                mem_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                return {
                    "utilization_pct": None,
                    "mem_used_gb": mem_used,
                    "mem_total_gb": mem_total,
                }
        except Exception:
            pass

        return None

    # ── CPU Stats ─────────────────────────────────────────────────

    @staticmethod
    def _get_cpu_stats() -> Optional[dict]:
        """Get CPU usage stats using psutil."""
        try:
            import psutil
            proc = psutil.Process()
            cpu_percent = proc.cpu_percent(interval=0)
            system_percent = psutil.cpu_percent(interval=0)
            thread_count = proc.num_threads()

            return {
                "process_pct": cpu_percent,
                "system_pct": system_percent,
                "thread_count": thread_count,
            }
        except Exception:
            return None

    # ── Report Generation ─────────────────────────────────────────

    def _print_report(self):
        """Generate and print the aggregate performance report (called under lock)."""
        elapsed = time.perf_counter() - self._report_start_time
        if elapsed <= 0:
            return

        # ── Helper functions ──
        def _avg(lst):
            return sum(lst) / len(lst) if lst else 0.0

        def _max(lst):
            return max(lst) if lst else 0.0

        def _min(lst):
            return min(lst) if lst else 0.0

        def _p95(lst):
            if not lst:
                return 0.0
            sorted_lst = sorted(lst)
            idx = int(len(sorted_lst) * 0.95)
            return sorted_lst[min(idx, len(sorted_lst) - 1)]

        def _sum(lst):
            return sum(lst) if lst else 0.0

        # ── Percentage calculation (sum/sum — correct for mixed sample counts) ──
        total_time_sum = _sum(self._timings.get("frame_total", []))

        def _pct_str(stage_name):
            """Calculate percentage as sum(stage_times) / sum(frame_total_times)."""
            stage_sum = _sum(self._timings.get(stage_name, []))
            if total_time_sum > 0:
                return f"({stage_sum / total_time_sum * 100:.0f}%)"
            return ""

        def _pct_val(stage_name):
            """Return percentage value for arithmetic."""
            stage_sum = _sum(self._timings.get(stage_name, []))
            if total_time_sum > 0:
                return stage_sum / total_time_sum * 100
            return 0.0

        # ── Amortized average (total time / total frames) ──
        def _amortized_avg(stage_name):
            """Average time per total frame (amortized over all frames, not just active ones)."""
            stage_sum = _sum(self._timings.get(stage_name, []))
            return stage_sum / self._total_frames if self._total_frames > 0 else 0.0

        # ── Stage formatting (Avg | P95) ──
        def _stage_line(label, stage_name):
            samples = self._timings.get(stage_name, [])
            if not samples:
                return None
            avg = _avg(samples)
            p95 = _p95(samples)
            pct = _pct_str(stage_name)
            return f"  {label:<16s}: Avg {avg:5.1f}ms | P95 {p95:5.1f}ms {pct}"

        # ── Build report ──
        lines = []
        lines.append("")
        lines.append("=" * 56)
        lines.append("  PERFORMANCE REPORT")
        lines.append("=" * 56)

        # ── Frame Summary ──
        lines.append("")
        lines.append(f"  Frames Processed : {self._total_frames}")
        lines.append(f"  Detection Frames : {self._detection_frames}")
        lines.append(f"  Report Period    : {elapsed:.1f}s")
        lines.append(f"  Slow Warnings    : {self._slow_warnings}")

        # ── FPS Breakdown ──
        lines.append("")
        lines.append("  ── FPS ──")
        pipeline_fps = self._total_frames / elapsed if elapsed > 0 else 0
        camera_fps = self._frames_received / elapsed if self._frames_received > 0 and elapsed > 0 else pipeline_fps
        lines.append(f"  Pipeline FPS     : {pipeline_fps:.1f}")
        lines.append(f"  Camera FPS       : {camera_fps:.1f}")
        lines.append(f"  Target FPS       : 30")

        # ── Camera Stats ──
        if self._frames_skipped > 0 or self._frames_dropped > 0:
            lines.append("")
            lines.append("  ── Camera ──")
            lines.append(f"  Received         : {self._frames_received}")
            lines.append(f"  Processed        : {self._total_frames}")
            lines.append(f"  Skipped (skip-N) : {self._frames_skipped}")
            lines.append(f"  Dropped          : {self._frames_dropped}")

        # ── Face Detection Stats ──
        if self._detection_frames > 0:
            avg_faces = self._faces_detected / self._detection_frames if self._detection_frames > 0 else 0
            lines.append("")
            lines.append("  ── Detection ──")
            lines.append(f"  Detection Frames : {self._detection_frames}")
            lines.append(f"  Faces Detected   : {self._faces_detected}")
            lines.append(f"  Avg Faces/Frame  : {avg_faces:.2f}")

        # ── SCRFD Sub-stage Timing ──
        scrfd_stages = [
            ("scrfd_preprocess", "Preprocess"),
            ("scrfd_inference", "Inference"),
            ("scrfd_postprocess", "Postprocess"),
        ]
        has_scrfd_substages = any(self._timings.get(s[0]) for s in scrfd_stages)
        if has_scrfd_substages:
            lines.append("")
            lines.append("  ── SCRFD Breakdown ──")
            for key, label in scrfd_stages:
                line = _stage_line(label, key)
                if line:
                    lines.append(line)
            # Total SCRFD (sum of sub-stages)
            scrfd_total_pct = sum(_pct_val(s[0]) for s in scrfd_stages)
            scrfd_total_avg = sum(_avg(self._timings.get(s[0], [])) for s in scrfd_stages)
            lines.append(f"  {'Total SCRFD':<16s}: Avg {scrfd_total_avg:5.1f}ms       ({scrfd_total_pct:.0f}%)")

        # ── AI Timing ──
        stages_ai = [
            ("scrfd_preprocess", "SCRFD Pre"),
            ("scrfd_inference", "SCRFD Infer"),
            ("scrfd_postprocess", "SCRFD Post"),
            ("arcface", "ArcFace"),
            ("matching", "Matching"),
        ]
        stages_non_ai = [
            ("bytetrack", "Tracker"),
            ("drawing", "Drawing"),
            ("jpeg_encode", "JPEG Encode"),
        ]

        ai_pct_total = sum(_pct_val(s[0]) for s in stages_ai)
        non_ai_pct_total = sum(_pct_val(s[0]) for s in stages_non_ai)

        lines.append("")
        lines.append("  ── AI Time ──")
        for key, label in stages_ai:
            line = _stage_line(label, key)
            if line:
                lines.append(line)
        lines.append(f"  {'Total AI':<16s}:                        ({ai_pct_total:.0f}%)")

        lines.append("")
        lines.append("  ── Non-AI Time ──")
        for key, label in stages_non_ai:
            line = _stage_line(label, key)
            if line:
                lines.append(line)
        lines.append(f"  {'Total Non-AI':<16s}:                        ({non_ai_pct_total:.0f}%)")

        # ── Frame Total ──
        lines.append("")
        other_pct = max(0, 100 - ai_pct_total - non_ai_pct_total)
        if other_pct > 1:
            lines.append(f"  {'Other':<16s}:                        ({other_pct:.0f}%)")
        ft_line = _stage_line("Frame Total", "frame_total")
        if ft_line:
            lines.append(ft_line)

        # ── GPU Stats ──
        gpu = self._get_gpu_stats()
        if gpu:
            lines.append("")
            lines.append("  ── GPU ──")
            if gpu.get("utilization_pct") is not None:
                lines.append(f"  Utilization      : {gpu['utilization_pct']}%")
            lines.append(f"  VRAM             : {gpu['mem_used_gb']:.1f} / {gpu['mem_total_gb']:.1f} GB")
            if gpu.get("temperature_c") is not None:
                lines.append(f"  Temperature      : {gpu['temperature_c']}°C")

        # ── CPU Stats ──
        cpu = self._get_cpu_stats()
        if cpu:
            lines.append("")
            lines.append("  ── CPU ──")
            lines.append(f"  Process          : {cpu['process_pct']:.1f}%")
            lines.append(f"  System           : {cpu['system_pct']:.1f}%")
            lines.append(f"  Threads          : {cpu['thread_count']}")

        # ── Recognition Cache ──
        if self._tracks_seen > 0:
            calls_saved = self._tracks_seen - self._recognition_calls
            hit_rate = (calls_saved / self._tracks_seen * 100) if self._tracks_seen > 0 else 0
            lines.append("")
            lines.append("  ── Recognition Cache ──")
            lines.append(f"  Tracks Seen      : {self._tracks_seen}")
            lines.append(f"  Recognition Calls: {self._recognition_calls}")
            lines.append(f"  Calls Saved      : {calls_saved}")
            lines.append(f"  Hit Rate         : {hit_rate:.1f}%")
            lines.append(f"  Tracks Recreated : {self._tracks_recreated}")
            lines.append(f"  Forced Rechecks  : {self._forced_rechecks}")

            # Cache reason breakdown
            if self._cache_reasons:
                lines.append("")
                lines.append("  Why recognized:")
                reason_labels = {
                    "no_identity": "No identity",
                    "new_track": "New track",
                    "recheck_unlocked": "Recheck unlock",
                    "recheck_locked": "Recheck locked",
                }
                total_reasons = sum(self._cache_reasons.values())
                for reason, count in sorted(self._cache_reasons.items(), key=lambda x: -x[1]):
                    label = reason_labels.get(reason, reason)
                    rpct = count / total_reasons * 100 if total_reasons > 0 else 0
                    lines.append(f"    {label:<18s}: {count:4d} ({rpct:.0f}%)")

            # Identity carryover stats
            carryover_total = self._carryover_success + self._carryover_failed
            if carryover_total > 0:
                carry_pct = self._carryover_success / carryover_total * 100
                lines.append("")
                lines.append(f"  Carry-over Success: {self._carryover_success}/{carryover_total} ({carry_pct:.0f}%)")
                lines.append(f"  Carry-over Failed : {self._carryover_failed}/{carryover_total} ({100 - carry_pct:.0f}%)")

        # ── Identity Continuity (Phase 1) ─────────────────────────
        if self._tracks_created > 0:
            recoverable_pct = self._tracks_recoverable / self._tracks_created * 100
            rate = self._tracks_created / self._detection_frames if self._detection_frames else 0.0
            lines.append("")
            lines.append("  ── Identity Continuity ──")
            lines.append(f"  Tracks Created   : {self._tracks_created}  ({rate:.2f}/detection-frame)")
            lines.append(
                f"  Recoverable      : {self._tracks_recoverable}/{self._tracks_created} "
                f"({recoverable_pct:.0f}%)"
            )
            lines.append(
                f"  Unrecoverable    : {self._tracks_created - self._tracks_recoverable} "
                f"(genuinely new person, or predecessor expired)"
            )
            if self._recovery_best_iou:
                lines.append(
                    f"  Best cand. IoU   : avg {_avg(self._recovery_best_iou):.3f} | "
                    f"p95 {_p95(self._recovery_best_iou):.3f} | max {_max(self._recovery_best_iou):.3f}"
                )
                lines.append(
                    f"  Best cand. dist  : avg {_avg(self._recovery_best_dist):.0f}px | "
                    f"p95 {_p95(self._recovery_best_dist):.0f}px"
                )
                lines.append(
                    f"  Recovery gap     : avg {_avg(self._recovery_gap_frames):.1f} frames | "
                    f"max {_max(self._recovery_gap_frames):.0f}"
                )
                # How many would a pure-IoU gate at 0.30 have caught?
                would_catch = sum(1 for v in self._recovery_best_iou if v > 0.30)
                lines.append(
                    f"  IoU>0.30 would   : {would_catch}/{len(self._recovery_best_iou)} "
                    f"({would_catch / len(self._recovery_best_iou) * 100:.0f}%) "
                    f"<- why a geometric-only gate is insufficient"
                )
            lines.append(f"  Identity Swaps   : {self._identity_swaps}")
            for ex in self._swap_examples:
                lines.append(f"    ! {ex}")

        # ── Event Queue ──
        if self._eq_processed > 0 or self._eq_dropped > 0:
            lines.append("")
            lines.append("  ── Event Queue ──")
            lines.append(f"  Processed        : {self._eq_processed}")
            lines.append(f"  Dropped          : {self._eq_dropped}")
            if self._eq_wait_times:
                lines.append(f"  Avg Wait Time    : {_avg(self._eq_wait_times):.1f} ms")
                lines.append(f"  Max Wait Time    : {_max(self._eq_wait_times):.1f} ms")

        # ── Snapshots ──
        if self._snapshots_saved > 0:
            lines.append("")
            lines.append("  ── Snapshots ──")
            lines.append(f"  Saved            : {self._snapshots_saved}")
            lines.append(f"  Avg Save Time    : {_avg(self._snapshot_times):.1f} ms")
            lines.append(f"  Max Save Time    : {_max(self._snapshot_times):.1f} ms")

        # ── Database ──
        if self._db_inserts > 0 or self._db_failures > 0:
            lines.append("")
            lines.append("  ── Database ──")
            lines.append(f"  Inserts          : {self._db_inserts}")
            lines.append(f"  Avg Insert Time  : {_avg(self._db_insert_times):.1f} ms")
            lines.append(f"  Max Insert Time  : {_max(self._db_insert_times):.1f} ms")
            lines.append(f"  Failed Inserts   : {self._db_failures}")

        # ── WebSocket / JPEG ──
        if self._ws_frame_sizes:
            avg_jpeg = _avg(self._ws_frame_sizes)
            lines.append("")
            lines.append("  ── WebSocket / JPEG ──")
            lines.append(f"  Avg JPEG Size    : {avg_jpeg / 1024:.1f} KB")
            lines.append(f"  Largest JPEG     : {_max(self._ws_frame_sizes) / 1024:.1f} KB")
            lines.append(f"  Smallest JPEG    : {_min(self._ws_frame_sizes) / 1024:.1f} KB")

            # Compression ratio
            if self._raw_frame_sizes:
                avg_raw = _avg(self._raw_frame_sizes)
                if avg_jpeg > 0:
                    ratio = avg_raw / avg_jpeg
                    lines.append(f"  Raw Frame Size   : {avg_raw / (1024 * 1024):.2f} MB")
                    lines.append(f"  Compression      : {ratio:.1f}:1")

            lines.append(f"  Dropped Frames   : {self._ws_dropped}")

        lines.append("")
        lines.append("=" * 56)
        lines.append("")

        report = "\n".join(lines)

        # Print to log
        logger.info(report)

        # Save to file
        if self._log_file:
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(self._log_file, "a", encoding="utf-8") as f:
                    f.write(f"[{timestamp}]\n{report}\n\n")
            except Exception:
                pass  # Don't crash the pipeline for log file errors

    # ── Startup Summary ───────────────────────────────────────────

    @staticmethod
    def print_startup_summary():
        """Print system info at server startup."""
        lines = []
        lines.append("")
        lines.append("═" * 56)
        lines.append("  SYSTEM INFO")
        lines.append("═" * 56)

        # GPU / CUDA / Torch
        try:
            import torch
            cuda_avail = torch.cuda.is_available()
            lines.append(f"  Torch Version    : {torch.__version__}")
            if cuda_avail:
                lines.append(f"  GPU Name         : {torch.cuda.get_device_name(0)}")
                lines.append(f"  CUDA Version     : {torch.version.cuda}")
                lines.append(f"  Inference Device : cuda:0")

                props = torch.cuda.get_device_properties(0)
                total_gb = props.total_memory / (1024 ** 3)
                lines.append(f"  GPU Memory       : {total_gb:.1f} GB")
            else:
                lines.append(f"  GPU Name         : N/A (CPU mode)")
                lines.append(f"  Inference Device : cpu")
        except ImportError:
            lines.append(f"  Torch            : Not installed")

        # ONNX Runtime providers
        try:
            import onnxruntime
            providers = onnxruntime.get_available_providers()
            lines.append(f"  ONNX Providers   : {', '.join(providers)}")
        except Exception:
            pass

        # JPEG encoder
        try:
            from app.utils.image_utils import get_jpeg_encoder_name
            jpeg_encoder = get_jpeg_encoder_name()
        except Exception:
            jpeg_encoder = "Unknown"
        lines.append(f"  JPEG Encoder     : {jpeg_encoder}")

        # Pipeline config
        from app.config import settings
        lines.append("")
        lines.append(f"  SCRFD Model      : {settings.INSIGHTFACE_MODEL_NAME}")
        lines.append(f"  ArcFace Model    : {settings.INSIGHTFACE_MODEL_NAME} (R50)")
        lines.append(f"  Pipeline Mode    : {settings.PIPELINE_MODE}")
        lines.append(f"  ByteTrack        : Enabled")
        lines.append(f"  Recog Cache      : Enabled (lock={settings.RECOGNITION_LOCK_CONFIDENCE}, "
                     f"recheck={settings.RECOGNITION_RECHECK_SEC}s/{settings.RECOGNITION_FULL_RECHECK_SEC}s)")
        lines.append(f"  Event Queue      : Enabled (size={settings.EVENT_QUEUE_SIZE})")
        lines.append(f"  Frame Skip       : {settings.FRAME_SKIP}")
        lines.append(f"  JPEG Quality     : 65")
        lines.append(f"  Stream Resolution: 960x540")
        lines.append(f"  Target FPS       : 30")
        lines.append(f"  Profiling        : {'Enabled' if settings.ENABLE_PROFILING else 'Disabled'}")

        lines.append("")
        lines.append("═" * 56)
        lines.append("")

        logger.info("\n".join(lines))


# ── Singleton ─────────────────────────────────────────────────────

def _create_profiler() -> PerformanceProfiler:
    """Create the singleton profiler with settings from config."""
    try:
        from app.config import settings
        enabled = settings.ENABLE_PROFILING
        log_file = os.path.join("logs", "performance.log")
    except Exception:
        enabled = True
        log_file = os.path.join("logs", "performance.log")

    p = PerformanceProfiler(report_interval=100, log_file=log_file)
    if not enabled:
        p.disable()
    return p


profiler = _create_profiler()
