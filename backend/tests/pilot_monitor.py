"""
Phase 4.2 -- Live Multi-Camera Global ID Pilot Monitor
=======================================================

Non-invasive observer that connects to the running SentinelAI backend
and monitors Global Identity Resolver decisions in real time.

Produces:
  - Real-time timeline log (CSV)
  - Cross-camera transition detection
  - Identity error detection (false merge, false split, drops)
  - Performance metrics
  - Summary report

Usage:
    cd backend && python tests/pilot_monitor.py [--duration 1800]

The monitor runs alongside the live pipeline, reading the resolver's
internal state and events without modifying any pipeline behaviour.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import signal
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# -- Config --
PILOT_DIR = Path(__file__).parent.parent / "validation" / "pilot"
LOG_FILE = PILOT_DIR / "pilot_timeline.csv"
REPORT_FILE = PILOT_DIR / "pilot_report.json"
SUMMARY_FILE = PILOT_DIR / "pilot_summary.txt"
SNAPSHOT_DIR = PILOT_DIR / "snapshots"

POLL_INTERVAL = 1.0  # seconds between resolver state polls
DEFAULT_DURATION = 1800  # 30 minutes

logger = logging.getLogger("pilot_monitor")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)


@dataclass
class TransitionRecord:
    """A detected cross-camera transition."""
    person_id: Optional[str]
    person_name: Optional[str]
    global_id: str
    camera_a: str
    camera_b: str
    time_a: float
    time_b: float
    track_a: Optional[int] = None
    track_b: Optional[int] = None
    travel_sec: float = 0.0
    verified: bool = False
    notes: str = ""


@dataclass
class PilotMetrics:
    """Accumulates metrics during the pilot run."""
    start_time: float = 0.0
    end_time: float = 0.0
    total_observations: int = 0
    confirmed: int = 0
    uncertain: int = 0
    new_ids: int = 0
    
    # Identity errors
    false_merges: int = 0
    false_splits: int = 0
    identity_drops: int = 0
    incorrect_reuse: int = 0
    track_rebinds: int = 0
    unknown_to_known: int = 0
    
    # Cross-camera
    transitions: List[TransitionRecord] = field(default_factory=list)
    
    # Person -> set of Global IDs seen
    person_global_ids: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    # Global ID -> set of persons seen
    global_id_persons: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    
    # Performance
    resolve_latencies: List[float] = field(default_factory=list)
    
    # Camera stats
    camera_observations: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Active state snapshots
    active_count_samples: List[Tuple[float, int]] = field(default_factory=list)
    missing_count_samples: List[Tuple[float, int]] = field(default_factory=list)
    expired_count_samples: List[Tuple[float, int]] = field(default_factory=list)


class PilotMonitor:
    """Monitors the GlobalIdentityResolver during a live pilot run."""
    
    def __init__(self, duration: float = DEFAULT_DURATION):
        self.duration = duration
        self.metrics = PilotMetrics()
        self._running = True
        self._last_event_idx = 0
        self._known_transitions: Dict[str, str] = {}  # global_id -> last camera
        self._csv_writer = None
        self._csv_file = None
        
    def _ensure_dirs(self):
        PILOT_DIR.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        
    def _open_csv(self):
        self._csv_file = open(LOG_FILE, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "timestamp", "camera", "local_track_id", "global_id",
            "known_person_id", "known_person_name", "state", "reason",
            "face_score", "body_score", "candidate_margin",
            "event_type", "details",
        ])
        
    def _close_csv(self):
        if self._csv_file:
            self._csv_file.flush()
            self._csv_file.close()
            
    def _log_event(self, event):
        """Log a single resolver event to CSV."""
        if not self._csv_writer:
            return
        ts = event.get("timestamp", time.time())
        state = event.get("state", {})
        details = event.get("details", {})
        self._csv_writer.writerow([
            datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
            state.get("last_camera", ""),
            state.get("local_track_id", ""),
            event.get("global_id", ""),
            state.get("known_person_id", ""),
            state.get("known_person_name", ""),
            event.get("lifecycle", ""),
            event.get("event_type", ""),
            details.get("face_similarity", ""),
            details.get("body_similarity", ""),
            details.get("candidate_margin", ""),
            event.get("event_type", ""),
            json.dumps(details) if details else "",
        ])
        self._csv_file.flush()
    
    def _detect_transitions(self, resolver):
        """Detect cross-camera transitions from active state."""
        with resolver._lock:
            for gid, state in resolver._active.items():
                last_cam = self._known_transitions.get(gid)
                current_cam = None
                current_track = None
                
                # Find the most recent camera from local bindings
                if state.local_bindings:
                    latest = max(state.local_bindings.items(), key=lambda x: x[1])
                    current_cam = latest[0][0]
                    current_track = latest[0][1]
                
                if current_cam and last_cam and current_cam != last_cam:
                    # Transition detected!
                    transition = TransitionRecord(
                        person_id=state.known_person_id,
                        person_name=state.known_person_name,
                        global_id=gid,
                        camera_a=last_cam,
                        camera_b=current_cam,
                        time_a=self._known_transitions.get(f"{gid}_time", time.time() - 5),
                        time_b=time.time(),
                        track_b=current_track,
                    )
                    transition.travel_sec = transition.time_b - transition.time_a
                    self.metrics.transitions.append(transition)
                    logger.info(
                        f"TRANSITION: {state.known_person_name or gid} "
                        f"{last_cam} -> {current_cam} "
                        f"(travel={transition.travel_sec:.1f}s)"
                    )
                
                if current_cam:
                    self._known_transitions[gid] = current_cam
                    self._known_transitions[f"{gid}_time"] = time.time()
    
    def _sample_state(self, resolver):
        """Take a snapshot of resolver state for metrics."""
        now = time.time()
        with resolver._lock:
            active = len(resolver._active)
            missing = len(resolver._missing)
            expired = len(resolver._expired)
        
        self.metrics.active_count_samples.append((now, active))
        self.metrics.missing_count_samples.append((now, missing))
        self.metrics.expired_count_samples.append((now, expired))
        
    def _process_events(self, resolver):
        """Process new events from the resolver."""
        events = resolver.events[self._last_event_idx:]
        self._last_event_idx = len(resolver.events)
        
        for event in events:
            event_dict = event.to_dict() if hasattr(event, "to_dict") else {}
            self._log_event(event_dict)
            self.metrics.total_observations += 1
            
            et = event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type)
            
            if "CREATED" in et:
                self.metrics.new_ids += 1
            elif "CONFIRMED" in et:
                self.metrics.confirmed += 1
            
            # Track person -> global_id mappings
            state = event.state if hasattr(event, "state") else None
            if state:
                gid = state.global_id if hasattr(state, "global_id") else None
                pid = state.known_person_id if hasattr(state, "known_person_id") else None
                cam = state.last_camera if hasattr(state, "last_camera") else None
                
                if gid and pid:
                    self.metrics.person_global_ids[pid].add(gid)
                    self.metrics.global_id_persons[gid].add(pid)
                
                if cam:
                    self.metrics.camera_observations[cam] += 1
    
    def _check_identity_errors(self):
        """Check for false merges and false splits."""
        self.metrics.false_merges = sum(
            1 for gid, persons in self.metrics.global_id_persons.items()
            if len(persons) > 1
        )
        self.metrics.false_splits = sum(
            1 for pid, gids in self.metrics.person_global_ids.items()
            if len(gids) > 1
        )
    
    def _generate_report(self):
        """Generate the final pilot report."""
        m = self.metrics
        duration = m.end_time - m.start_time
        
        report = {
            "pilot_duration_sec": round(duration, 1),
            "pilot_duration_human": f"{int(duration // 60)}m {int(duration % 60)}s",
            "cameras": dict(m.camera_observations),
            "total_observations": m.total_observations,
            "confirmed": m.confirmed,
            "uncertain": m.uncertain,
            "new_global_ids": m.new_ids,
            "unique_persons": len(m.person_global_ids),
            "unique_global_ids": len(m.global_id_persons),
            "cross_camera_transitions": len(m.transitions),
            "identity_errors": {
                "false_merges": m.false_merges,
                "false_splits": m.false_splits,
                "identity_drops": m.identity_drops,
                "incorrect_reuse": m.incorrect_reuse,
            },
            "person_global_ids": {k: list(v) for k, v in m.person_global_ids.items()},
            "global_id_persons": {k: list(v) for k, v in m.global_id_persons.items()},
            "transitions": [
                {
                    "person": t.person_name or t.person_id or "unknown",
                    "global_id": t.global_id,
                    "from_camera": t.camera_a,
                    "to_camera": t.camera_b,
                    "travel_sec": round(t.travel_sec, 1),
                    "time_a": datetime.fromtimestamp(t.time_a, tz=timezone.utc).isoformat(),
                    "time_b": datetime.fromtimestamp(t.time_b, tz=timezone.utc).isoformat(),
                }
                for t in m.transitions
            ],
            "state_samples": {
                "active_max": max((s[1] for s in m.active_count_samples), default=0),
                "missing_max": max((s[1] for s in m.missing_count_samples), default=0),
                "expired_max": max((s[1] for s in m.expired_count_samples), default=0),
                "samples_count": len(m.active_count_samples),
            },
        }
        
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        
        # Human-readable summary
        lines = [
            "=" * 80,
            "PHASE 4.2 -- LIVE PILOT SUMMARY",
            "=" * 80,
            "",
            f"Duration:             {report['pilot_duration_human']}",
            f"Cameras:              {list(report['cameras'].keys())}",
            f"Total observations:   {report['total_observations']}",
            f"Confirmed:            {report['confirmed']}",
            f"New Global IDs:       {report['new_global_ids']}",
            f"Unique persons:       {report['unique_persons']}",
            f"Unique Global IDs:    {report['unique_global_ids']}",
            "",
            "CROSS-CAMERA TRANSITIONS",
            f"  Total transitions:  {report['cross_camera_transitions']}",
        ]
        
        for i, t in enumerate(report["transitions"], 1):
            lines.append(f"  [{i}] {t['person']} : {t['from_camera']} -> {t['to_camera']} ({t['travel_sec']}s)")
        
        lines += [
            "",
            "IDENTITY ERRORS",
            f"  False merges:       {m.false_merges}",
            f"  False splits:       {m.false_splits}",
            f"  Identity drops:     {m.identity_drops}",
            f"  Incorrect reuse:    {m.incorrect_reuse}",
        ]
        
        if m.false_merges > 0:
            lines.append("  FALSE MERGE DETAILS:")
            for gid, persons in m.global_id_persons.items():
                if len(persons) > 1:
                    lines.append(f"    Global ID {gid} -> persons: {persons}")
        
        if m.false_splits > 0:
            lines.append("  FALSE SPLIT DETAILS:")
            for pid, gids in m.person_global_ids.items():
                if len(gids) > 1:
                    lines.append(f"    Person {pid} -> Global IDs: {gids}")
        
        lines += [
            "",
            "STATE PEAKS",
            f"  Active max:         {report['state_samples']['active_max']}",
            f"  Missing max:        {report['state_samples']['missing_max']}",
            f"  Expired max:        {report['state_samples']['expired_max']}",
            "",
            "PER-CAMERA OBSERVATIONS",
        ]
        for cam, count in sorted(report["cameras"].items()):
            lines.append(f"  {cam}: {count}")
        
        lines += ["", "=" * 80]
        
        summary = "\n".join(lines)
        with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
            f.write(summary)
        print(summary)
        
        return report
    
    def run(self):
        """Run the pilot monitor."""
        self._ensure_dirs()
        self._open_csv()
        
        logger.info(f"Starting pilot monitor (duration={self.duration}s)")
        logger.info(f"Logs: {LOG_FILE}")
        logger.info(f"Report: {REPORT_FILE}")
        
        # Get the resolver singleton
        try:
            from app.vision.global_identity import get_global_resolver
            resolver = get_global_resolver()
        except Exception as e:
            logger.error(f"Cannot get resolver: {e}")
            logger.info("The backend must be running with GLOBAL_ID_MODE=shadow or authoritative")
            self._close_csv()
            return None
        
        if resolver is None:
            logger.error("Resolver is None -- GLOBAL_ID_MODE may be 'off'")
            self._close_csv()
            return None
        
        logger.info(f"Resolver connected (mode={resolver.mode.value})")
        
        self.metrics.start_time = time.time()
        deadline = self.metrics.start_time + self.duration
        
        # Signal handler for graceful shutdown
        def _stop(sig, frame):
            logger.info("Pilot monitor stopping (signal received)...")
            self._running = False
        signal.signal(signal.SIGINT, _stop)
        
        poll_count = 0
        while self._running and time.time() < deadline:
            try:
                self._process_events(resolver)
                self._detect_transitions(resolver)
                self._sample_state(resolver)
                self._check_identity_errors()
                
                poll_count += 1
                if poll_count % 30 == 0:  # Status every 30 seconds
                    elapsed = time.time() - self.metrics.start_time
                    remaining = deadline - time.time()
                    logger.info(
                        f"Pilot: {elapsed:.0f}s elapsed, {remaining:.0f}s remaining, "
                        f"observations={self.metrics.total_observations}, "
                        f"transitions={len(self.metrics.transitions)}, "
                        f"merges={self.metrics.false_merges}, "
                        f"splits={self.metrics.false_splits}"
                    )
                
                time.sleep(POLL_INTERVAL)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                time.sleep(POLL_INTERVAL)
        
        self.metrics.end_time = time.time()
        
        # Final event sweep
        self._process_events(resolver)
        self._check_identity_errors()
        
        self._close_csv()
        report = self._generate_report()
        
        logger.info(f"Pilot complete. Report: {REPORT_FILE}")
        return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 4.2 Live Pilot Monitor")
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION,
                       help="Pilot duration in seconds (default: 1800 = 30 min)")
    args = parser.parse_args()
    
    monitor = PilotMonitor(duration=args.duration)
    monitor.run()


if __name__ == "__main__":
    main()
