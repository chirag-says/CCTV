"""
Person Tracker — Entry/Exit Algorithm.

Implements time-based entry/exit tracking with:
- Configurable thresholds
- Cooldown periods to prevent rapid re-entry
- State machine per tracked person
- Thread-safe operation
"""

import threading
import time
import logging
from typing import Dict, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class TrackState:
    """State of a tracked person."""
    person_id: str
    person_name: str
    camera_id: str
    first_seen: float  # Unix timestamp
    last_seen: float
    detection_count: int = 1
    entry_confirmed: bool = False
    exit_emitted: bool = False
    session_id: Optional[str] = None

    @property
    def duration(self) -> float:
        """Duration from first to last detection in seconds."""
        return self.last_seen - self.first_seen

    @property
    def time_since_last_seen(self) -> float:
        """Seconds since last detection."""
        return time.time() - self.last_seen


class PersonTracker:
    """
    Tracks persons' entry and exit based on face detection events.
    
    Algorithm:
    1. First detection → start tracking (not yet confirmed)
    2. After ENTRY_THRESHOLD seconds of detections → confirm ENTRY
    3. If not detected for EXIT_THRESHOLD seconds → emit EXIT
    4. After EXIT, a COOLDOWN prevents immediate re-entry counting
    
    Thread-safe for concurrent camera pipelines.
    """

    def __init__(
        self,
        entry_threshold: int = None,
        exit_threshold: int = None,
        cooldown_period: int = 10,
        on_entry: Optional[Callable] = None,
        on_exit: Optional[Callable] = None,
    ):
        self.entry_threshold = entry_threshold or settings.ENTRY_THRESHOLD_SECONDS
        self.exit_threshold = exit_threshold or settings.EXIT_THRESHOLD_SECONDS
        self.cooldown_period = cooldown_period

        # Callbacks
        self.on_entry = on_entry
        self.on_exit = on_exit

        # Active tracks: person_id -> TrackState
        self._active_tracks: Dict[str, TrackState] = {}
        # Cooldown: person_id -> timestamp when cooldown expires
        self._cooldowns: Dict[str, float] = {}
        self._lock = threading.RLock()

        # Statistics
        self._total_entries = 0
        self._total_exits = 0

        logger.info(
            f"PersonTracker initialized: entry={self.entry_threshold}s, "
            f"exit={self.exit_threshold}s, cooldown={self.cooldown_period}s"
        )

    def on_detection(
        self,
        person_id: str,
        person_name: str,
        camera_id: str,
        confidence: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> Optional[dict]:
        """
        Process a face detection event.
        
        Args:
            person_id: Unique person identifier
            person_name: Display name
            camera_id: Camera that detected the person
            confidence: Match confidence
            timestamp: Detection time (defaults to now)
            
        Returns:
            Event dict if entry was confirmed, else None
        """
        ts = timestamp or time.time()

        with self._lock:
            # Check cooldown
            if person_id in self._cooldowns:
                if ts < self._cooldowns[person_id]:
                    return None  # Still in cooldown
                else:
                    del self._cooldowns[person_id]

            if person_id in self._active_tracks:
                track = self._active_tracks[person_id]
                track.last_seen = ts
                track.detection_count += 1

                # Update camera if person moved
                if track.camera_id != camera_id:
                    track.camera_id = camera_id

                # Check entry confirmation
                if not track.entry_confirmed:
                    if track.duration >= self.entry_threshold:
                        track.entry_confirmed = True
                        self._total_entries += 1

                        entry_event = {
                            "event_type": "entry",
                            "person_id": person_id,
                            "person_name": person_name,
                            "camera_id": camera_id,
                            "confidence": confidence,
                            "timestamp": datetime.fromtimestamp(
                                track.first_seen, tz=timezone.utc
                            ).isoformat(),
                            "detection_count": track.detection_count,
                        }

                        logger.info(f"ENTRY confirmed: {person_name} (camera: {camera_id})")

                        if self.on_entry:
                            self.on_entry(entry_event)

                        return entry_event

                return None
            else:
                # New person spotted — start tracking
                self._active_tracks[person_id] = TrackState(
                    person_id=person_id,
                    person_name=person_name,
                    camera_id=camera_id,
                    first_seen=ts,
                    last_seen=ts,
                )

                # Check immediate entry (threshold=0)
                if self.entry_threshold <= 0:
                    track = self._active_tracks[person_id]
                    track.entry_confirmed = True
                    self._total_entries += 1

                    entry_event = {
                        "event_type": "entry",
                        "person_id": person_id,
                        "person_name": person_name,
                        "camera_id": camera_id,
                        "confidence": confidence,
                        "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                        "detection_count": 1,
                    }

                    if self.on_entry:
                        self.on_entry(entry_event)

                    return entry_event

                return None

    def check_exits(self, current_time: Optional[float] = None) -> List[dict]:
        """
        Check for persons who should be marked as exited.
        
        Should be called periodically (e.g., every 10 seconds).
        
        Returns:
            List of exit event dicts
        """
        now = current_time or time.time()
        exit_events = []

        with self._lock:
            exited_ids = []

            for person_id, track in self._active_tracks.items():
                absence_duration = now - track.last_seen

                if absence_duration >= self.exit_threshold:
                    if track.entry_confirmed:
                        exit_event = {
                            "event_type": "exit",
                            "person_id": track.person_id,
                            "person_name": track.person_name,
                            "camera_id": track.camera_id,
                            "timestamp": datetime.fromtimestamp(
                                track.last_seen, tz=timezone.utc
                            ).isoformat(),
                            "entry_time": datetime.fromtimestamp(
                                track.first_seen, tz=timezone.utc
                            ).isoformat(),
                            "duration_sec": int(track.last_seen - track.first_seen),
                            "detection_count": track.detection_count,
                            "session_id": track.session_id,
                        }
                        exit_events.append(exit_event)
                        self._total_exits += 1

                        logger.info(
                            f"EXIT detected: {track.person_name} "
                            f"(duration: {exit_event['duration_sec']}s)"
                        )

                        if self.on_exit:
                            self.on_exit(exit_event)

                    exited_ids.append(person_id)

            # Remove exited tracks and set cooldowns
            for pid in exited_ids:
                del self._active_tracks[pid]
                self._cooldowns[pid] = now + self.cooldown_period

        return exit_events

    def get_active_persons(self) -> List[dict]:
        """Get list of currently tracked (present) persons."""
        with self._lock:
            return [
                {
                    "person_id": t.person_id,
                    "person_name": t.person_name,
                    "camera_id": t.camera_id,
                    "entry_confirmed": t.entry_confirmed,
                    "first_seen": datetime.fromtimestamp(
                        t.first_seen, tz=timezone.utc
                    ).isoformat(),
                    "last_seen": datetime.fromtimestamp(
                        t.last_seen, tz=timezone.utc
                    ).isoformat(),
                    "duration_sec": int(t.duration),
                    "detection_count": t.detection_count,
                }
                for t in self._active_tracks.values()
                if t.entry_confirmed
            ]

    def is_person_present(self, person_id: str) -> bool:
        """Check if a person is currently tracked as present."""
        with self._lock:
            track = self._active_tracks.get(person_id)
            return track is not None and track.entry_confirmed

    def get_occupancy(self) -> int:
        """Get current occupancy count (confirmed entries only)."""
        with self._lock:
            return sum(
                1 for t in self._active_tracks.values() if t.entry_confirmed
            )

    def set_session_id(self, person_id: str, session_id: str):
        """Associate a database session ID with an active track."""
        with self._lock:
            if person_id in self._active_tracks:
                self._active_tracks[person_id].session_id = session_id

    def reset(self):
        """Clear all tracking state."""
        with self._lock:
            self._active_tracks.clear()
            self._cooldowns.clear()
            logger.info("Tracker state reset")

    @property
    def stats(self) -> dict:
        return {
            "active_tracks": len(self._active_tracks),
            "confirmed_present": self.get_occupancy(),
            "total_entries": self._total_entries,
            "total_exits": self._total_exits,
            "cooldowns_active": len(self._cooldowns),
        }
