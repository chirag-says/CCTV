"""
Pipeline Benchmark — Lightweight per-stage latency instrumentation.

When enabled via ENABLE_PIPELINE_BENCHMARKS, measures and logs the time
taken by each stage of the vision pipeline. Logs a summary every 100
detection frames.

Usage in pipeline:
    from app.vision.benchmarks import pipeline_bench

    with pipeline_bench.measure("scrfd_detect"):
        face_locations, embeddings, rgb_frame = self.detector.detect_faces_with_embeddings(frame)

    with pipeline_bench.measure("arcface_match"):
        result = self.recognizer.recognize_face(embedding)
"""

import time
import logging
import threading
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List

from app.config import settings

logger = logging.getLogger(__name__)


class PipelineBenchmark:
    """
    Collects per-stage timing data and logs summaries periodically.

    Thread-safe: each camera worker has its own measurements but they
    are aggregated into a shared log output.
    """

    def __init__(self, log_interval: int = 100):
        self._enabled = settings.ENABLE_PIPELINE_BENCHMARKS
        self._log_interval = log_interval
        self._lock = threading.Lock()

        # stage_name → list of durations (ms)
        self._measurements: Dict[str, List[float]] = defaultdict(list)
        self._frame_count = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @contextmanager
    def measure(self, stage_name: str):
        """
        Context manager to measure a pipeline stage.

        Usage:
            with bench.measure("scrfd_detect"):
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
                self._measurements[stage_name].append(elapsed_ms)

    def tick_frame(self):
        """
        Called once per detection frame. When the interval is reached,
        logs a summary and resets.
        """
        if not self._enabled:
            return

        with self._lock:
            self._frame_count += 1
            if self._frame_count >= self._log_interval:
                self._log_summary()
                self._frame_count = 0
                self._measurements.clear()

    def _log_summary(self):
        """Log timing summary (called under lock)."""
        if not self._measurements:
            return

        total_avg = 0.0

        # Stages in display order with friendly labels
        stage_config = [
            ("scrfd_detect",      "SCRFD detect"),
            ("arcface_match",     "ArcFace"),
            ("bytetrack_update",  "ByteTrack"),
            ("queue_put",         "Queue.put()"),
            ("jpeg_encode",       "JPEG encode"),
            ("frame_total",       "Frame total"),
        ]

        # Log known stages in order, then any extra
        known_keys = {s[0] for s in stage_config}
        extra_stages = [
            (s, s) for s in self._measurements if s not in known_keys
        ]
        all_stages = list(stage_config) + extra_stages

        lines = []
        for key, label in all_stages:
            samples = self._measurements.get(key)
            if not samples:
                continue
            avg_ms = sum(samples) / len(samples)
            max_ms = max(samples)
            min_ms = min(samples)
            if key != "frame_total":
                total_avg += avg_ms
            lines.append(
                f"  {label:<16s} avg {avg_ms:6.1f}ms  "
                f"min {min_ms:6.1f}ms  max {max_ms:6.1f}ms  "
                f"(n={len(samples)})"
            )

        header = f"[BENCH] Pipeline timing ({self._log_interval} detection frames):"
        footer = f"  {'Pipeline cost':<16s} avg {total_avg:6.1f}ms (excl. frame_total)"
        body = "\n".join(lines)
        logger.info(f"{header}\n{body}\n{footer}")


# -- Singleton -----------------------------------------------------------------
pipeline_bench = PipelineBenchmark(log_interval=30)
