"""
Stream Health Monitor — Detects frozen/dead RTSP streams and manages auto-recovery.

Tracks stream liveness via:
  - Frame read success/failure
  - Frame content hash (detects frozen streams that still read() == True)
  - Monotonic timestamps for timeout enforcement

State machine:
  LIVE → STALE → RECONNECTING → LIVE (recovered)
                                → OFFLINE (max attempts exceeded)

Every state transition is logged and emits a security alert event.
"""

import enum
import logging
import time
import zlib
from datetime import datetime, timezone
from typing import Callable, Optional

import cv2
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class StreamState(str, enum.Enum):
    """Explicit stream health states — never use boolean flags."""
    LIVE = "LIVE"
    STALE = "STALE"
    RECONNECTING = "RECONNECTING"
    OFFLINE = "OFFLINE"


class StreamHealthMonitor:
    """
    Monitors stream health and manages auto-recovery for a single camera.

    Sits inside the grabber loop. On every frame read, call `on_frame_read()`
    with the result. On read failure, call `on_read_failure()`.

    The monitor handles:
    - Timeout-based staleness detection (no fresh frame for N seconds)
    - Frozen-frame detection (identical CRC32 hash for N seconds)
    - State transitions with structured logging
    - Security alert emission through the existing event system

    This class is thread-safe for single-writer (grabber thread) usage.
    The `state` and `status` properties can be read from any thread.
    """

    def __init__(
        self,
        camera_id: str,
        on_event: Optional[Callable] = None,
        stale_timeout: float = None,
        reconnect_initial: float = None,
        reconnect_max: float = None,
        reconnect_backoff: float = None,
        frozen_hash_timeout: float = None,
        max_reconnect_attempts: int = None,
    ):
        self.camera_id = camera_id
        self._on_event = on_event

        # Configuration (fall back to settings)
        self._stale_timeout = stale_timeout or settings.STREAM_STALE_TIMEOUT_SEC
        self._reconnect_initial = reconnect_initial or settings.STREAM_RECONNECT_INITIAL_SEC
        self._reconnect_max = reconnect_max or settings.STREAM_RECONNECT_MAX_SEC
        self._reconnect_backoff = reconnect_backoff or settings.STREAM_RECONNECT_BACKOFF
        self._frozen_hash_timeout = frozen_hash_timeout or settings.STREAM_FROZEN_HASH_TIMEOUT_SEC
        self._max_reconnect_attempts = max_reconnect_attempts or settings.STREAM_MAX_RECONNECT_ATTEMPTS

        # State
        self._state: StreamState = StreamState.LIVE
        self._last_new_frame_monotonic: float = time.monotonic()
        self._last_new_frame_utc: datetime = datetime.now(timezone.utc)
        self._stale_since: Optional[datetime] = None
        self._reconnect_attempts: int = 0
        self._consecutive_failures: int = 0

        # Frozen-frame detection
        self._last_frame_hash: Optional[int] = None
        self._same_hash_since: float = time.monotonic()

        # Reconnect backoff state
        self._current_backoff: float = self._reconnect_initial
        self._last_reconnect_attempt: float = 0.0

        logger.info(
            f"[{camera_id}] StreamHealthMonitor initialized: "
            f"stale_timeout={self._stale_timeout}s, "
            f"frozen_hash_timeout={self._frozen_hash_timeout}s, "
            f"reconnect_backoff={self._reconnect_initial}s→{self._reconnect_max}s"
        )

    @property
    def state(self) -> StreamState:
        """Current stream state. Safe to read from any thread."""
        return self._state

    @property
    def stale_since(self) -> Optional[datetime]:
        return self._stale_since

    @property
    def reconnect_attempts(self) -> int:
        return self._reconnect_attempts

    @property
    def last_frame_age(self) -> float:
        """Seconds since last fresh frame (monotonic, not wall-clock)."""
        return time.monotonic() - self._last_new_frame_monotonic

    @property
    def last_frame_time(self) -> Optional[str]:
        """ISO-formatted UTC time of last fresh frame."""
        return self._last_new_frame_utc.isoformat()

    @property
    def status(self) -> dict:
        """Full health status dict for API responses."""
        return {
            "stream_state": self._state.value,
            "stale_since": self._stale_since.isoformat() if self._stale_since else None,
            "reconnect_attempts": self._reconnect_attempts,
            "last_frame_age": round(self.last_frame_age, 1),
            "last_frame_time": self.last_frame_time,
        }

    def on_frame_read(self, frame: np.ndarray) -> bool:
        """
        Called by the grabber loop after a successful read().

        Checks for frozen frames (identical content hash). Returns True if
        the frame is genuinely new (different content), False if it's a
        duplicate of the previous frame.

        On a genuinely new frame, transitions back to LIVE if currently STALE
        or RECONNECTING.
        """
        # Compute lightweight frame hash (CRC32 of downsampled frame)
        frame_hash = self._compute_frame_hash(frame)

        if frame_hash == self._last_frame_hash:
            # Same content as previous frame — may be frozen
            elapsed = time.monotonic() - self._same_hash_since
            if elapsed >= self._frozen_hash_timeout and self._state == StreamState.LIVE:
                logger.warning(
                    f"[{self.camera_id}] Frozen stream detected: "
                    f"identical frame hash for {elapsed:.1f}s"
                )
                self._transition_to_stale("frozen_frame")
            return False
        else:
            # Genuinely new frame
            self._last_frame_hash = frame_hash
            self._same_hash_since = time.monotonic()
            self._last_new_frame_monotonic = time.monotonic()
            self._last_new_frame_utc = datetime.now(timezone.utc)
            self._consecutive_failures = 0

            if self._state != StreamState.LIVE:
                self._transition_to_live()
            return True

    def on_read_failure(self):
        """
        Called by the grabber loop when read() returns False.

        Increments failure counter and transitions to STALE if timeout exceeded.
        """
        self._consecutive_failures += 1

        if self._state == StreamState.LIVE:
            age = self.last_frame_age
            if age >= self._stale_timeout:
                logger.warning(
                    f"[{self.camera_id}] Stream read failures: "
                    f"{self._consecutive_failures} consecutive, "
                    f"last frame {age:.1f}s ago"
                )
                self._transition_to_stale("read_failure")

    def check_staleness(self):
        """
        Periodic check (call from grabber loop even when reads succeed).
        Catches edge case where reads succeed but no new content arrives.
        """
        if self._state == StreamState.LIVE:
            age = self.last_frame_age
            if age >= self._stale_timeout:
                self._transition_to_stale("timeout")

    def should_reconnect(self) -> bool:
        """
        Check if it's time to attempt a reconnection.

        Returns True if enough backoff time has elapsed since the last attempt.
        The caller (grabber loop) should release and reopen the capture.
        """
        if self._state not in (StreamState.STALE, StreamState.RECONNECTING):
            return False

        if self._reconnect_attempts >= self._max_reconnect_attempts:
            if self._state != StreamState.OFFLINE:
                self._transition_to_offline()
            return False

        now = time.monotonic()
        if now - self._last_reconnect_attempt >= self._current_backoff:
            return True

        return False

    def on_reconnect_attempt(self):
        """Called when the grabber loop begins a reconnection attempt."""
        self._reconnect_attempts += 1
        self._last_reconnect_attempt = time.monotonic()

        if self._state != StreamState.RECONNECTING:
            self._state = StreamState.RECONNECTING
            logger.info(
                f"[{self.camera_id}] Stream state: RECONNECTING "
                f"(attempt #{self._reconnect_attempts})"
            )

        # Exponential backoff
        self._current_backoff = min(
            self._current_backoff * self._reconnect_backoff,
            self._reconnect_max,
        )

        logger.info(
            f"[{self.camera_id}] Reconnect attempt #{self._reconnect_attempts} "
            f"(next backoff: {self._current_backoff:.1f}s)"
        )

    def on_reconnect_success(self):
        """Called when capture.isOpened() returns True after reconnection."""
        # Reset backoff for next time
        self._current_backoff = self._reconnect_initial
        # Frame hash will be reset naturally on next new frame
        self._last_frame_hash = None
        # Don't transition to LIVE yet — wait for actual frame content
        logger.info(
            f"[{self.camera_id}] Capture reopened successfully, "
            f"waiting for first frame..."
        )

    def reset(self):
        """Reset all state (e.g., on manual camera restart)."""
        self._state = StreamState.LIVE
        self._last_new_frame_monotonic = time.monotonic()
        self._last_new_frame_utc = datetime.now(timezone.utc)
        self._stale_since = None
        self._reconnect_attempts = 0
        self._consecutive_failures = 0
        self._last_frame_hash = None
        self._same_hash_since = time.monotonic()
        self._current_backoff = self._reconnect_initial
        self._last_reconnect_attempt = 0.0

    # ── Private ─────────────────────────────────────────────────

    def _compute_frame_hash(self, frame: np.ndarray) -> int:
        """
        Lightweight frame hash for frozen-frame detection.

        Downsamples to 64x36 (tiny) and computes CRC32. This is NOT
        cryptographic — it's a fast content-change detector. CRC32 of
        a 6912-byte buffer takes ~1µs, which is negligible.
        """
        try:
            small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_NEAREST)
            return zlib.crc32(small.tobytes())
        except Exception:
            # Fallback: hash raw bytes (more expensive but reliable)
            return zlib.crc32(frame.tobytes()[:4096])

    def _transition_to_stale(self, reason: str):
        """Transition from LIVE to STALE."""
        prev = self._state
        self._state = StreamState.STALE
        self._stale_since = datetime.now(timezone.utc)

        logger.warning(
            f"[{self.camera_id}] Stream state: {prev.value} → STALE "
            f"(reason={reason}, last_frame_age={self.last_frame_age:.1f}s)"
        )

        self._emit_stream_event("stream_offline", {
            "stream_state": StreamState.STALE.value,
            "reason": reason,
            "last_frame_age": round(self.last_frame_age, 1),
        })

    def _transition_to_live(self):
        """Transition from STALE/RECONNECTING to LIVE."""
        prev = self._state
        downtime = (
            (datetime.now(timezone.utc) - self._stale_since).total_seconds()
            if self._stale_since else 0.0
        )
        attempts = self._reconnect_attempts

        self._state = StreamState.LIVE
        self._stale_since = None
        self._reconnect_attempts = 0
        self._consecutive_failures = 0
        self._current_backoff = self._reconnect_initial

        logger.info(
            f"[{self.camera_id}] Stream state: {prev.value} → LIVE "
            f"(recovered after {downtime:.1f}s, {attempts} reconnect attempts)"
        )

        self._emit_stream_event("stream_recovered", {
            "stream_state": StreamState.LIVE.value,
            "downtime_sec": round(downtime, 1),
            "reconnect_attempts": attempts,
        })

    def _transition_to_offline(self):
        """Transition to OFFLINE after max reconnect attempts."""
        self._state = StreamState.OFFLINE

        logger.error(
            f"[{self.camera_id}] Stream state: OFFLINE "
            f"(gave up after {self._reconnect_attempts} reconnect attempts)"
        )

        self._emit_stream_event("stream_offline", {
            "stream_state": StreamState.OFFLINE.value,
            "reason": "max_reconnect_attempts",
            "reconnect_attempts": self._reconnect_attempts,
        })

    def _emit_stream_event(self, subtype: str, metadata: dict):
        """Emit a security alert event through the existing event system."""
        if self._on_event is None:
            return

        event = {
            "event_type": "security_alert",
            "subtype": subtype,
            "camera_id": self.camera_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "confidence": 1.0,
            "metadata": metadata,
        }

        try:
            self._on_event(event)
        except Exception as e:
            logger.error(
                f"[{self.camera_id}] Failed to emit stream health event: {e}"
            )
