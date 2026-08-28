"""
Person Tracker — Entry/Exit Algorithm.

Implements time-based entry/exit tracking with:
- Configurable thresholds
- Cooldown periods to prevent rapid re-entry
- State machine per tracked person
- Centroid history for loitering / crowd analytics
- Thread-safe operation
"""

import threading
import time
import logging
from collections import deque
from typing import Dict, Optional, List, Tuple, Callable
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
    # Centroid history for loitering/crowd analysis: deque of (timestamp, cx, cy)
    centroid_history: deque = field(default_factory=lambda: deque(maxlen=600))
    last_centroid: Optional[Tuple[int, int]] = None

    @property
    def duration(self) -> float:
        """Duration from first to last detection in seconds."""
        return self.last_seen - self.first_seen

    @property
    def time_since_last_seen(self) -> float:
        """Seconds since last detection."""
        return time.time() - self.last_seen


@dataclass
class UnidentifiedState:
    """
    Someone the cameras can see but have never identified.

    Keyed by the integer ByteTrack id, NOT by a synthetic string identity.
    The `anon:<camera>:<track>` handle this replaces was a string pretending
    to be a person: it entered `_active_tracks` and `_cooldowns` beside real
    identities, and because it embedded the track id it minted a brand-new
    "person" every time ByteTrack recreated the track.
    """
    track_id: int
    camera_id: str
    first_seen: float
    last_seen: float
    detection_count: int = 1
    entry_confirmed: bool = False
    last_centroid: Optional[Tuple[int, int]] = None

    @property
    def duration(self) -> float:
        return self.last_seen - self.first_seen


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
        cooldown_period: int = 60,  # Prevents re-entry detection for 60s after exit
        on_entry: Optional[Callable] = None,
        on_exit: Optional[Callable] = None,
        on_unidentified_entry: Optional[Callable] = None,
        on_unidentified_exit: Optional[Callable] = None,
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

        # -- Unidentified occupants (Step 1.5) --------------------
        # A SEPARATE lane, keyed by ByteTrack id. Nothing here ever reaches
        # `_active_tracks`, so attendance, presence and journeys keep meaning
        # exactly what they meant before: things that happened to a person we
        # can name.
        self.on_unidentified_entry = on_unidentified_entry
        self.on_unidentified_exit = on_unidentified_exit
        self._unidentified: Dict[int, UnidentifiedState] = {}
        self._unidentified_entry_sec = getattr(settings, "UNIDENTIFIED_ENTRY_SEC", 5.0)
        self._unidentified_exit_sec = getattr(settings, "UNIDENTIFIED_EXIT_SEC", 15.0)
        self._unidentified_max = getattr(settings, "UNIDENTIFIED_MAX_TRACKS", 200)
        self._total_unidentified = 0

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
        centroid: Optional[Tuple[int, int]] = None,
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

                # Update centroid history
                if centroid is not None:
                    track.last_centroid = centroid
                    track.centroid_history.append((ts, centroid[0], centroid[1]))

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
                new_track = TrackState(
                    person_id=person_id,
                    person_name=person_name,
                    camera_id=camera_id,
                    first_seen=ts,
                    last_seen=ts,
                )
                # Store initial centroid
                if centroid is not None:
                    new_track.last_centroid = centroid
                    new_track.centroid_history.append((ts, centroid[0], centroid[1]))
                self._active_tracks[person_id] = new_track

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

            self._prune_cooldowns(now)

        return exit_events

    def _prune_cooldowns(self, now: float):
        """
        Drop cooldowns that have expired.

        Without this the map only ever shrank when that EXACT person_id was
        seen again (on_detection) or on a full reset. For enrolled people that
        is self-limiting -- there are only so many of them. For any id that
        never repeats it is a slow leak, which is precisely what the old
        `anon:<camera>:<track>` handles were: a fresh key every time ByteTrack
        recreated a track. Step 1.5 removes that emitter; this makes the
        structure safe regardless of what feeds it.
        """
        if not self._cooldowns:
            return
        expired = [pid for pid, until in self._cooldowns.items() if now >= until]
        for pid in expired:
            del self._cooldowns[pid]

    # -- Unidentified occupants -----------------------------------

    def on_unidentified(
        self,
        track_id: int,
        camera_id: str,
        centroid: Optional[Tuple[int, int]] = None,
        timestamp: Optional[float] = None,
    ) -> Optional[dict]:
        """
        Record a sighting of a tracked person we cannot name.

        Counted, never named. This deliberately does NOT touch
        `_active_tracks`: everything downstream of that dict -- attendance,
        presence, journeys, campus summary -- answers questions about a
        specific person, and an unidentified body cannot answer them.

        Returns an occupancy-start event once the track has been visible for
        UNIDENTIFIED_ENTRY_SEC, else None.
        """
        ts = timestamp or time.time()

        with self._lock:
            state = self._unidentified.get(track_id)
            if state is None:
                if len(self._unidentified) >= self._unidentified_max:
                    # Bounded by construction. Evict the stalest rather than
                    # refusing, so a camera watching a crowd keeps reporting
                    # the people currently in front of it.
                    oldest = min(self._unidentified.values(), key=lambda s: s.last_seen)
                    del self._unidentified[oldest.track_id]
                self._unidentified[track_id] = UnidentifiedState(
                    track_id=track_id,
                    camera_id=camera_id,
                    first_seen=ts,
                    last_seen=ts,
                    last_centroid=centroid,
                )
                return None

            state.last_seen = ts
            state.detection_count += 1
            state.camera_id = camera_id
            if centroid is not None:
                state.last_centroid = centroid

            if state.entry_confirmed:
                return None
            if state.duration < self._unidentified_entry_sec:
                return None

            state.entry_confirmed = True
            self._total_unidentified += 1
            event = self._unidentified_event(state, "unidentified_entry", state.first_seen)

        # Callback outside the lock -- it reaches the event queue.
        if self.on_unidentified_entry:
            self.on_unidentified_entry(event)
        return event

    @staticmethod
    def _unidentified_event(state: "UnidentifiedState", subtype: str, ts: float) -> dict:
        """
        Shape an occupancy event.

        event_type is 'detection', never 'entry'/'exit': every attendance,
        journey and analytics query filters on those two, and an unidentified
        body must not be able to inflate footfall. The subtype carries the
        real meaning.
        """
        return {
            "event_type": "detection",
            "subtype": subtype,
            "camera_id": state.camera_id,
            "confidence": 0.0,
            "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
            "track_id": state.track_id,
            "detection_count": state.detection_count,
            "duration_sec": int(state.last_seen - state.first_seen),
            "unidentified": True,
        }

    def expire_unidentified(self, current_time: Optional[float] = None) -> List[dict]:
        """
        Close out unidentified tracks that have not been seen recently.

        Time-based, so the map cannot grow without bound however badly the
        tracker fragments.
        """
        now = current_time or time.time()
        events = []

        with self._lock:
            gone = [
                tid for tid, st in self._unidentified.items()
                if now - st.last_seen >= self._unidentified_exit_sec
            ]
            for tid in gone:
                state = self._unidentified.pop(tid)
                if state.entry_confirmed:
                    events.append(
                        self._unidentified_event(
                            state, "unidentified_exit", state.last_seen
                        )
                    )

        for event in events:
            if self.on_unidentified_exit:
                self.on_unidentified_exit(event)
        return events

    def get_unidentified_centroids(self) -> Dict[str, Tuple[int, int]]:
        """
        {"track:<id>": (cx, cy)} for unidentified occupants.

        The key is namespaced so it can be merged with get_person_centroids()
        for crowd density -- which counts bodies -- without ever colliding
        with a real person_id.
        """
        with self._lock:
            return {
                f"track:{tid}": st.last_centroid
                for tid, st in self._unidentified.items()
                if st.last_centroid is not None
            }

    @property
    def unidentified_count(self) -> int:
        """Unidentified people currently considered present."""
        with self._lock:
            return sum(1 for st in self._unidentified.values() if st.entry_confirmed)

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

    def get_person_centroids(self) -> Dict[str, Tuple[int, int]]:
        """
        Get current centroid position for all actively tracked persons.
        Used by crowd/loitering detectors.

        Returns:
            {person_id: (cx, cy)} for persons with a known centroid
        """
        with self._lock:
            return {
                pid: track.last_centroid
                for pid, track in self._active_tracks.items()
                if track.last_centroid is not None
            }

    def get_active_track_times(self) -> Dict[str, float]:
        """
        Get last detection time for all active tracks.
        Used by the gathering detector to filter stale tracks.

        Returns:
            {person_id: last_seen_unix_timestamp}
        """
        with self._lock:
            return {
                pid: track.last_seen
                for pid, track in self._active_tracks.items()
            }

    def get_person_names(self) -> Dict[str, str]:
        """Get {person_id: person_name} for all active tracks."""
        with self._lock:
            return {
                pid: track.person_name
                for pid, track in self._active_tracks.items()
            }

    def reset(self):
        """Clear all tracking state."""
        with self._lock:
            self._active_tracks.clear()
            self._cooldowns.clear()
            self._unidentified.clear()
            logger.info("Tracker state reset")

    @property
    def stats(self) -> dict:
        return {
            "active_tracks": len(self._active_tracks),
            "confirmed_present": self.get_occupancy(),
            "total_entries": self._total_entries,
            "total_exits": self._total_exits,
            "cooldowns_active": len(self._cooldowns),
            "unidentified_present": self.unidentified_count,
            "unidentified_tracked": len(self._unidentified),
            "total_unidentified": self._total_unidentified,
        }
