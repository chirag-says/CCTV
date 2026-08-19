"""
Async Snapshot Worker — Priority-based background snapshot writer.

Decouples snapshot disk I/O from the pipeline/event processing threads.
Three priority tiers:
  - CRITICAL (0): Unknown face evidence — never dropped
  - NORMAL  (1): Known person updates — dropped if queue > 80%
  - DEBUG   (2): Debug/diagnostic frames — dropped if queue > 50%

Thread-safe, singleton, auto-starts on first submit.
"""

import os
import queue
import threading
import time
import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Tuple, Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SnapshotPriority(IntEnum):
    """Lower number = higher priority."""
    CRITICAL = 0  # Unknown evidence — never dropped
    NORMAL = 1    # Known person crops — dropped under pressure
    DEBUG = 2     # Diagnostic frames — dropped aggressively


@dataclass(order=True)
class SnapshotJob:
    """A prioritised snapshot write job."""
    priority: int
    # Fields below are not used for ordering
    image: np.ndarray = field(compare=False)
    name: str = field(compare=False)
    subdirectory: str = field(compare=False, default="")
    quality: int = field(compare=False, default=90)
    callback: Optional[Callable[[str, str], None]] = field(compare=False, default=None)
    submitted_at: float = field(compare=False, default_factory=time.monotonic)
    # Tie-breaker: FIFO order within same priority
    seq: int = field(compare=True, default=0)


class SnapshotWorker:
    """
    Background thread that writes snapshots with priority ordering.

    Usage:
        worker = get_snapshot_worker()
        worker.submit(image, "face_abc123", "unknowns", 95,
                      priority=SnapshotPriority.CRITICAL,
                      callback=lambda fp, url: save_url_to_db(url))
    """

    _instance: Optional["SnapshotWorker"] = None
    _lock = threading.Lock()

    def __init__(self, max_queue: int = 200):
        self._queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=max_queue)
        self._max_queue = max_queue
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._seq = 0  # Monotonic counter for FIFO within same priority
        self._seq_lock = threading.Lock()

        # Stats
        self._processed = 0
        self._dropped_normal = 0
        self._dropped_debug = 0
        self._total_write_ms = 0.0
        self._max_write_ms = 0.0
        self._started = False

        # simplejpeg availability (reuse the detection from image_utils)
        self._simplejpeg = None
        self._simplejpeg_available = False
        try:
            import simplejpeg
            self._simplejpeg = simplejpeg
            self._simplejpeg_available = True
        except ImportError:
            pass

    def _next_seq(self) -> int:
        with self._seq_lock:
            self._seq += 1
            return self._seq

    def start(self):
        """Start the background writer thread."""
        if self._started:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="SnapshotWorker", daemon=True
        )
        self._thread.start()
        self._started = True
        logger.info("📸 SnapshotWorker started")

    def stop(self):
        """Signal the worker to stop and drain remaining jobs."""
        if not self._started:
            return
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._started = False
        logger.info(
            f"📸 SnapshotWorker stopped — processed={self._processed}, "
            f"dropped_normal={self._dropped_normal}, dropped_debug={self._dropped_debug}"
        )

    def submit(
        self,
        image: np.ndarray,
        name: str,
        subdirectory: str = "",
        quality: int = 90,
        priority: SnapshotPriority = SnapshotPriority.NORMAL,
        callback: Optional[Callable[[str, str], None]] = None,
    ) -> bool:
        """
        Submit a snapshot job. Returns True if accepted, False if dropped.

        Drop policy:
          - CRITICAL: never dropped (blocks if queue is full)
          - NORMAL: dropped if queue > 80% capacity
          - DEBUG: dropped if queue > 50% capacity
        """
        if not self._started:
            self.start()

        current_size = self._queue.qsize()
        fill_pct = current_size / self._max_queue

        # Drop policy
        if priority == SnapshotPriority.DEBUG and fill_pct > 0.5:
            self._dropped_debug += 1
            return False
        if priority == SnapshotPriority.NORMAL and fill_pct > 0.8:
            self._dropped_normal += 1
            return False

        job = SnapshotJob(
            priority=int(priority),
            seq=self._next_seq(),
            image=image,
            name=name,
            subdirectory=subdirectory,
            quality=quality,
            callback=callback,
        )

        if priority == SnapshotPriority.CRITICAL:
            # Critical: block until space available (never drop evidence)
            self._queue.put(job, block=True, timeout=10.0)
        else:
            try:
                self._queue.put_nowait(job)
            except queue.Full:
                if priority == SnapshotPriority.NORMAL:
                    self._dropped_normal += 1
                else:
                    self._dropped_debug += 1
                return False

        return True

    def _run(self):
        """Worker loop: pull jobs from priority queue and write them."""
        while not self._stop_event.is_set():
            try:
                job: SnapshotJob = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                filepath, url_path = self._write_snapshot(job)
                if job.callback and filepath:
                    job.callback(filepath, url_path)
            except Exception as e:
                logger.error(f"SnapshotWorker write failed: {e}")
            finally:
                self._queue.task_done()

        # Drain remaining jobs on shutdown
        while not self._queue.empty():
            try:
                job = self._queue.get_nowait()
                filepath, url_path = self._write_snapshot(job)
                if job.callback and filepath:
                    job.callback(filepath, url_path)
                self._queue.task_done()
            except Exception:
                break

    def _write_snapshot(self, job: SnapshotJob) -> Tuple[str, str]:
        """Write a single snapshot to disk with fsync."""
        start = time.perf_counter()

        base_dir = os.path.join("snapshots", job.subdirectory) if job.subdirectory else "snapshots"
        os.makedirs(base_dir, exist_ok=True)

        filename = f"{job.name}.jpg"
        filepath = os.path.join(base_dir, filename)

        if self._simplejpeg_available:
            rgb = cv2.cvtColor(job.image, cv2.COLOR_BGR2RGB)
            jpeg_bytes = self._simplejpeg.encode_jpeg(
                rgb, quality=job.quality, colorspace='RGB'
            )
            with open(filepath, 'wb') as f:
                f.write(jpeg_bytes)
                f.flush()
                os.fsync(f.fileno())
        else:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, job.quality]
            cv2.imwrite(filepath, job.image, encode_params)
            try:
                fd = os.open(filepath, os.O_RDONLY)
                os.fsync(fd)
                os.close(fd)
            except OSError:
                pass

        url_parts = ["snapshots"]
        if job.subdirectory:
            url_parts.append(job.subdirectory)
        url_parts.append(filename)
        url_path = "/" + "/".join(url_parts)

        elapsed_ms = (time.perf_counter() - start) * 1000
        self._processed += 1
        self._total_write_ms += elapsed_ms
        self._max_write_ms = max(self._max_write_ms, elapsed_ms)

        latency_ms = (time.monotonic() - job.submitted_at) * 1000
        if latency_ms > 500:
            logger.warning(
                f"📸 Snapshot queue latency: {latency_ms:.0f}ms for {filepath} "
                f"(priority={job.priority})"
            )

        return filepath, url_path

    @property
    def stats(self) -> dict:
        return {
            "queue_size": self._queue.qsize(),
            "max_queue": self._max_queue,
            "fill_pct": round(self._queue.qsize() / self._max_queue * 100, 1),
            "processed": self._processed,
            "dropped_normal": self._dropped_normal,
            "dropped_debug": self._dropped_debug,
            "avg_write_ms": round(
                self._total_write_ms / self._processed, 1
            ) if self._processed else 0,
            "max_write_ms": round(self._max_write_ms, 1),
            "running": self._started,
        }


# ── Singleton ─────────────────────────────────────────────────────

_worker_instance: Optional[SnapshotWorker] = None
_worker_lock = threading.Lock()


def get_snapshot_worker(max_queue: int = 200) -> SnapshotWorker:
    """Get or create the singleton SnapshotWorker."""
    global _worker_instance
    if _worker_instance is None:
        with _worker_lock:
            if _worker_instance is None:
                _worker_instance = SnapshotWorker(max_queue=max_queue)
                _worker_instance.start()
    return _worker_instance
