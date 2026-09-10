"""
Load Governor — one process-wide view of how hard the vision stack is working.

THE GAP THIS FILLS
`VisionPipeline._apply_load_degradation` tiers on the face count of ITS OWN
camera. That was written when a deployment meant one or two cameras, where
"this camera is busy" and "the machine is busy" were the same statement.

With 13 pipelines they are not. Each camera sees about one face, so every one
of them concludes "normal operation" and keeps its full detection rate, while
collectively they oversubscribe the GPU and the GIL by roughly 3x. Measured
before this existed: SCRFD p95 of 108.8 ms against an uncontended cost of
9.6-15 ms, with the GPU sitting at 22% utilisation -- idle, while every camera
waited on the others.

No camera can see that from its own face count. Something has to look at the
whole process.

WHY LATENCY, NOT A CAPACITY MODEL
The obvious alternative is to compute a budget: cameras x detection rate x
cost per inference. Every term in that is a guess that goes stale -- a
different model, a changed det_size, another process on the same GPU, a
thermal cap. Observed detection latency measures the thing we actually care
about, needs no calibration, and is already on the hot path.

So this is a closed loop: hold p95 detection latency near a target by handing
every pipeline the same additional frame skip. When latency is fine the bonus
decays to zero and the governor is invisible.

WHY SKIPPING FRAMES IS SAFE NOW, AND WAS NOT BEFORE
Before person tracking, a skipped detection frame meant a dropped track: a
face not detected was a person who ceased to exist. Since Step 1 the tracker
follows bodies and coasts through gaps, so a lower detection rate costs
recognition FRESHNESS -- how soon a face is re-checked -- rather than track
continuity. That is what makes this an acceptable lever at all.

THREADING
One instance per process, touched from every camera thread. `record()` is on
the hot path and takes a lock only to append to a deque, which is O(1).
"""

import logging
import threading
import time
from collections import deque
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)


class LoadGovernor:
    """Holds detection latency near a target by throttling every pipeline."""

    def __init__(
        self,
        target_ms: float = None,
        max_extra_skip: int = None,
        adjust_interval: float = None,
        window: int = None,
    ):
        self.target_ms = target_ms if target_ms is not None else getattr(
            settings, "LOAD_TARGET_DETECT_MS", 60.0)
        self.max_extra_skip = max_extra_skip if max_extra_skip is not None else getattr(
            settings, "LOAD_MAX_EXTRA_SKIP", 6)
        self.adjust_interval = adjust_interval if adjust_interval is not None else getattr(
            settings, "LOAD_ADJUST_INTERVAL_SEC", 5.0)
        window = window if window is not None else getattr(
            settings, "LOAD_SAMPLE_WINDOW", 300)

        self._samples: deque = deque(maxlen=window)
        self._lock = threading.Lock()
        self._extra_skip = 0
        self._last_adjust = time.time()
        self._p95 = 0.0
        self._cameras: dict = {}

        # Counters for the health endpoint.
        self.stat_raised = 0
        self.stat_lowered = 0

    # ── Hot path ─────────────────────────────────────────────────────────

    def record(self, duration_ms: float, camera_id: str = "") -> None:
        """Report one detection pass. Called from every camera thread."""
        now = time.time()
        with self._lock:
            self._samples.append(duration_ms)
            if camera_id:
                self._cameras[camera_id] = now

    def extra_skip(self) -> int:
        """
        Additional frames to skip, on top of the camera's own FRAME_SKIP.

        Cheap by design: the adjustment itself runs at most once per
        `adjust_interval`, and every other call is a dict read.
        """
        if not getattr(settings, "LOAD_GOVERNOR_ENABLED", True):
            return 0
        now = time.time()
        if now - self._last_adjust >= self.adjust_interval:
            self._adjust(now)
        return self._extra_skip

    # ── Control loop ─────────────────────────────────────────────────────

    def _adjust(self, now: float) -> None:
        with self._lock:
            # Re-check inside the lock: several camera threads reach this at
            # once, and only one of them should move the setting.
            if now - self._last_adjust < self.adjust_interval:
                return
            self._last_adjust = now

            if len(self._samples) < 20:
                return  # too little evidence to act on

            ordered = sorted(self._samples)
            p95 = ordered[int(len(ordered) * 0.95) - 1]
            self._p95 = p95

            # Drop cameras that have gone quiet, so the active count stays honest.
            self._cameras = {
                c: t for c, t in self._cameras.items() if now - t < 30.0
            }

            before = self._extra_skip
            # A deliberately wide dead band. Detection latency is noisy, and a
            # governor that chases every fluctuation oscillates -- which costs
            # more than the load it is trying to shed.
            if p95 > self.target_ms * 1.2 and self._extra_skip < self.max_extra_skip:
                self._extra_skip += 1
                self.stat_raised += 1
            elif p95 < self.target_ms * 0.7 and self._extra_skip > 0:
                self._extra_skip -= 1
                self.stat_lowered += 1

            if self._extra_skip != before:
                logger.info(
                    f"LoadGovernor: detection p95 {p95:.0f} ms vs target "
                    f"{self.target_ms:.0f} ms across {len(self._cameras)} camera(s) "
                    f"-> extra frame skip {before} -> {self._extra_skip}"
                )
            self._samples.clear()

    # ── Introspection ────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "detect_p95_ms": round(self._p95, 1),
                "target_ms": self.target_ms,
                "extra_frame_skip": self._extra_skip,
                "active_cameras": len(self._cameras),
                "samples": len(self._samples),
                "raised": self.stat_raised,
                "lowered": self.stat_lowered,
            }

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._cameras.clear()
            self._extra_skip = 0
            self._p95 = 0.0
            self._last_adjust = time.time()


_governor: Optional[LoadGovernor] = None
_governor_lock = threading.Lock()


def get_load_governor() -> LoadGovernor:
    """Process-wide singleton. One machine, one view of its load."""
    global _governor
    if _governor is None:
        with _governor_lock:
            if _governor is None:
                _governor = LoadGovernor()
    return _governor
