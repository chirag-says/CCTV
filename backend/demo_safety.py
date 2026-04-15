"""
🎬 Safety & Security — LIVE VISUAL DEMO
========================================
Run:  python demo_safety.py

Opens your webcam and runs ALL 4 safety modules with REDUCED thresholds
so you can see them trigger quickly during testing.

HOW TO TEST EACH FEATURE:
─────────────────────────
1. ATTRIBUTE RECOGNITION  → Just stand in front of the camera.
   Look at the console — it will print your detected gender + clothing color.

2. CROWD DETECTION        → Get 2 friends in frame, stand close together.
   (Or hold up a phone showing a photo of people!)
   Threshold reduced to 2 people + 3 seconds for easy testing.

3. LOITERING DETECTION    → Stand still for ~15 seconds (reduced from 5 min).
   A "LOITERING" warning will appear on screen.

4. HAZARD DETECTION       → Hold up a backpack, knife, or handbag in frame.
   YOLOv8 will detect it and show a red bounding box.

KEYBOARD CONTROLS:
  Q / ESC  → Quit
  S        → Show current stats
  R        → Reset all detectors
"""

import cv2
import numpy as np
import time
import sys
import os
import threading

# Add the backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.vision.safety_analytics import (
    AttributeRecognizer,
    CrowdDetector,
    LoiteringDetector,
    _dominant_clothing_color,
)
from app.vision.hazard_detector import HazardDetector

# ── Configuration (reduced thresholds for demo) ──────────────────────────────
DEMO_CONFIG = {
    "crowd_proximity_px": 200,     # Pixels (more lenient)
    "crowd_min_persons": 2,        # 2 people instead of 3
    "crowd_sustain_sec": 3.0,      # 3 seconds instead of 5
    "loiter_movement_px": 60,      # Pixels
    "loiter_time_sec": 15.0,       # 15 seconds instead of 5 minutes!
    "hazard_frame_interval": 10,   # Every 10th frame (more frequent for demo)
}

# ── Alert state (shared across callbacks) ─────────────────────────────────────
class AlertState:
    def __init__(self):
        self.lock = threading.Lock()
        self.active_alerts: list = []  # Recent alerts to display on screen
        self.alert_log: list = []      # Full history
        self.attributes: dict = {}     # person_id → attributes

    def add_alert(self, alert: dict):
        with self.lock:
            alert["display_until"] = time.time() + 8  # Show for 8 seconds
            self.active_alerts.append(alert)
            self.alert_log.append(alert)
            subtype = alert.get("subtype", "unknown")
            print(f"\n  🚨 ALERT: {subtype.upper()}")
            for k, v in alert.items():
                if k not in ("display_until", "metadata"):
                    print(f"     {k}: {v}")

    def get_active(self) -> list:
        with self.lock:
            now = time.time()
            self.active_alerts = [a for a in self.active_alerts if a["display_until"] > now]
            return list(self.active_alerts)

    def set_attributes(self, track_id: str, attrs: dict):
        with self.lock:
            self.attributes[track_id] = attrs
            print(f"\n  👤 ATTRIBUTES for {track_id[:8]}…")
            print(f"     Gender:   {attrs.get('gender', '?')}")
            print(f"     Apparel:  {attrs.get('apparel', '?')}")
            print(f"     Color:    {attrs.get('clothing_color', '?')}")


state = AlertState()


def on_alert(alert):
    state.add_alert(alert)


def on_attributes(track_id, attrs):
    state.set_attributes(track_id, attrs)


# ── Initialize modules ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  🎬  SAFETY & SECURITY — LIVE DEMO")
print("=" * 60)
print("\nInitializing modules...")

attr_recognizer = AttributeRecognizer(max_workers=2)

crowd_detector = CrowdDetector(
    proximity_px=DEMO_CONFIG["crowd_proximity_px"],
    min_persons=DEMO_CONFIG["crowd_min_persons"],
    sustain_seconds=DEMO_CONFIG["crowd_sustain_sec"],
    on_alert=on_alert,
)

loiter_detector = LoiteringDetector(
    movement_threshold_px=DEMO_CONFIG["loiter_movement_px"],
    time_window_sec=DEMO_CONFIG["loiter_time_sec"],
    alert_cooldown_sec=30.0,
    on_alert=on_alert,
)

hazard_detector = HazardDetector(
    frame_interval=DEMO_CONFIG["hazard_frame_interval"],
    alert_cooldown_sec=10.0,
    on_alert=on_alert,
)

print("✅ All modules initialized")
print(f"\nDemo thresholds (reduced for testing):")
for k, v in DEMO_CONFIG.items():
    print(f"  {k}: {v}")

# -- Face detection (using SCRFD via insightface) --------
try:
    from app.vision.detector import FaceDetector
    _demo_detector = FaceDetector()
    FACE_DETECTION_AVAILABLE = True
    print("OK  Face detection available (SCRFD via insightface)")
except Exception as e:
    FACE_DETECTION_AVAILABLE = False
    print(f"!!  SCRFD not available ({e}) -- using Haar cascade fallback")
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )


def detect_faces(frame):
    """Detect faces and return list of (top, right, bottom, left) + centroids."""
    if FACE_DETECTION_AVAILABLE:
        locations, _ = _demo_detector.detect_faces(frame)
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
        locations = [(y, x+w, y+h, x) for (x, y, w, h) in faces]

    centroids = {}
    for i, (t, r, b, l) in enumerate(locations):
        cx, cy = (l + r) // 2, (t + b) // 2
        centroids[f"person_{i}"] = (cx, cy)

    return locations, centroids


# ── Main loop ─────────────────────────────────────────────────────────────────
print("\n🎥 Opening webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam!")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("✅ Webcam opened")
print("\n" + "─" * 60)
print("  CONTROLS:  Q/ESC=Quit  S=Stats  R=Reset")
print("─" * 60)
print("\n  Stand in front of the camera to start testing!\n")

frame_count = 0
fps_start = time.time()
fps_counter = 0
fps_display = 0.0
last_safety_check = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        now = time.time()
        clean_frame = frame.copy()

        # ── FPS calculation ──
        fps_counter += 1
        if now - fps_start >= 1.0:
            fps_display = fps_counter / (now - fps_start)
            fps_counter = 0
            fps_start = now

        # ── HAZARD DETECTION (independent, every Nth frame) ──
        hazard_detector.submit_frame(clean_frame, "demo-cam")

        # ── FACE DETECTION (every 3rd frame) ──
        if frame_count % 3 == 0:
            face_locations, person_centroids = detect_faces(frame)

            # Draw face boxes
            for i, (top, right, bottom, left) in enumerate(face_locations):
                pid = f"person_{i}"
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # Attribute label
                attrs = state.attributes.get(pid)
                if attrs:
                    label = f"{attrs.get('gender','?')} | {attrs.get('apparel','?')}"
                else:
                    label = f"Person {i}"

                cv2.putText(frame, label, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Run attribute recognition (once per person)
                attr_recognizer.maybe_extract(
                    track_id=pid,
                    frame=clean_frame,
                    face_location=(top, right, bottom, left),
                    on_complete=on_attributes,
                )

            # ── SAFETY ANALYTICS (every 1 second) ──
            if now - last_safety_check >= 1.0 and person_centroids:
                last_safety_check = now
                person_names = {pid: pid for pid in person_centroids}

                # Crowd detection
                crowd_detector.update(person_centroids, "demo-cam")

                # Loitering detection
                loiter_detector.update(person_centroids, person_names, "demo-cam")

            # Show person count
            count_text = f"Persons: {len(face_locations)}"
            cv2.putText(frame, count_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # ── Draw active alerts on frame ──
        alerts = state.get_active()
        y_offset = 90
        for alert in alerts:
            subtype = alert.get("subtype", "?").upper()
            if subtype == "GATHERING":
                color = (0, 0, 255)
                text = f"!! CROWD GATHERING DETECTED ({alert.get('person_count',0)} people) !!"
            elif subtype == "LOITERING":
                color = (0, 165, 255)
                text = f"!! LOITERING: {alert.get('person_name','?')} !!"
            elif subtype == "HAZARD":
                color = (0, 0, 255)
                text = f"!! HAZARD: {alert.get('threat_class','?').upper()} !!"
            else:
                color = (0, 255, 255)
                text = f"!! ALERT: {subtype} !!"

            # Flashing effect
            if int(now * 3) % 2 == 0:
                cv2.putText(frame, text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 30

        # ── Status bar ──
        cv2.putText(frame, f"FPS: {fps_display:.0f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        modules_text = "ATTR | CROWD | LOITER | HAZARD"
        cv2.putText(frame, modules_text, (frame.shape[1] - 320, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # ── Instructions overlay ──
        instructions = [
            "Q=Quit | S=Stats | R=Reset",
            "Stand still 15s = Loitering alert",
            "Hold backpack/knife = Hazard alert",
        ]
        for i, inst in enumerate(instructions):
            cv2.putText(frame, inst, (10, frame.shape[0] - 10 - (len(instructions) - 1 - i) * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # ── Display ──
        cv2.imshow("Safety & Security Demo - Press Q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # Q or ESC
            break
        elif key == ord('s'):
            print("\n" + "=" * 40)
            print("  📊 CURRENT STATS")
            print("=" * 40)
            print(f"  Frames processed: {frame_count}")
            print(f"  FPS: {fps_display:.1f}")
            print(f"  Hazard detector: {hazard_detector.stats}")
            print(f"  Total alerts: {len(state.alert_log)}")
            print(f"  Attributes cached: {len(state.attributes)}")
            for pid, attrs in state.attributes.items():
                print(f"    {pid}: {attrs}")
            print("=" * 40 + "\n")
        elif key == ord('r'):
            attr_recognizer.reset()
            crowd_detector.reset()
            loiter_detector.reset()
            state.active_alerts.clear()
            state.alert_log.clear()
            state.attributes.clear()
            print("\n  🔄 All detectors reset!\n")

except KeyboardInterrupt:
    print("\n\nInterrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    attr_recognizer.shutdown()
    hazard_detector.shutdown()

    print("\n" + "=" * 60)
    print("  📊 SESSION SUMMARY")
    print("=" * 60)
    print(f"  Total frames: {frame_count}")
    print(f"  Total alerts: {len(state.alert_log)}")
    for alert in state.alert_log:
        subtype = alert.get("subtype", "?")
        ts = alert.get("timestamp", "?")
        print(f"    [{ts}] {subtype}")
    print(f"  Persons analyzed: {len(state.attributes)}")
    for pid, attrs in state.attributes.items():
        print(f"    {pid}: {attrs}")
    print("=" * 60 + "\n")
