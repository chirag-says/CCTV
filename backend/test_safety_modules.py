"""
Safety & Security Modules — Validation Test Script
===================================================
Run with:  python test_safety_modules.py

Tests all 4 modules with synthetic data (no camera/DB/GPU needed).
"""

import sys
import time
import numpy as np

PASS = "\033[92m✔ PASS\033[0m"
FAIL = "\033[91m✘ FAIL\033[0m"
WARN = "\033[93m⚠ WARN\033[0m"
results = []


def test(name, fn):
    """Run a test and record result."""
    try:
        fn()
        print(f"  {PASS}  {name}")
        results.append(True)
    except Exception as e:
        print(f"  {FAIL}  {name}  →  {e}")
        results.append(False)


# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  SAFETY & SECURITY MODULES — VALIDATION")
print("=" * 60)

# ── 1. IMPORTS ────────────────────────────────────────────────────────────────
print("\n📦 1. Import Checks")

def test_import_safety_analytics():
    from app.vision.safety_analytics import (
        AttributeRecognizer, CrowdDetector, LoiteringDetector,
        _dominant_clothing_color, _analyze_gender,
    )

def test_import_hazard_detector():
    from app.vision.hazard_detector import HazardDetector, THREAT_CLASSES

def test_import_pipeline():
    from app.vision.pipeline import VisionPipeline

def test_import_tracker():
    from app.vision.tracker import PersonTracker, TrackState

def test_import_config():
    from app.config import settings
    assert hasattr(settings, "ENABLE_ATTRIBUTE_RECOGNITION")
    assert hasattr(settings, "ENABLE_CROWD_DETECTION")
    assert hasattr(settings, "ENABLE_LOITERING_DETECTION")
    assert hasattr(settings, "ENABLE_HAZARD_DETECTION")
    assert hasattr(settings, "CROWD_PROXIMITY_PX")
    assert hasattr(settings, "LOITER_MOVEMENT_THRESHOLD_PX")
    assert hasattr(settings, "HAZARD_MODEL_PATH")

test("Import safety_analytics", test_import_safety_analytics)
test("Import hazard_detector", test_import_hazard_detector)
test("Import pipeline", test_import_pipeline)
test("Import tracker (with centroids)", test_import_tracker)
test("Import config (new settings)", test_import_config)


# ── 2. CLOTHING COLOR DETECTION ──────────────────────────────────────────────
print("\n🎨 2. Clothing Color Detection (HSV)")

def test_clothing_red():
    from app.vision.safety_analytics import _dominant_clothing_color
    # Create a fake 200x200 frame with a red "torso" area below a face
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    # Fill torso region (below face box) with red (BGR: 0, 0, 255)
    frame[60:150, 40:160] = (0, 0, 255)
    location = (10, 160, 55, 40)  # (top, right, bottom, left) — face box
    color = _dominant_clothing_color(frame, location)
    assert color == "Red", f"Expected Red, got {color}"

def test_clothing_blue():
    from app.vision.safety_analytics import _dominant_clothing_color
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[60:150, 40:160] = (255, 100, 0)  # BGR blue
    location = (10, 160, 55, 40)
    color = _dominant_clothing_color(frame, location)
    assert color == "Blue", f"Expected Blue, got {color}"

def test_clothing_black():
    from app.vision.safety_analytics import _dominant_clothing_color
    frame = np.zeros((200, 200, 3), dtype=np.uint8)  # All black
    location = (10, 160, 55, 40)
    color = _dominant_clothing_color(frame, location)
    assert color == "Black", f"Expected Black, got {color}"

test("Detect red clothing", test_clothing_red)
test("Detect blue clothing", test_clothing_blue)
test("Detect black clothing", test_clothing_black)


# ── 3. ATTRIBUTE RECOGNIZER ──────────────────────────────────────────────────
print("\n👤 3. Attribute Recognizer")

def test_attr_recognizer_once_per_track():
    from app.vision.safety_analytics import AttributeRecognizer
    ar = AttributeRecognizer(max_workers=1)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    location = (10, 160, 55, 40)
    # First call — should submit
    result1 = ar.maybe_extract("track-001", frame, location)
    # Second call with same ID — should skip (return cached or None)
    result2 = ar.maybe_extract("track-001", frame, location)
    # Wait for background thread (DeepFace first-load can be slow)
    attrs = None
    for _ in range(30):
        attrs = ar.get_attributes("track-001")
        if attrs is not None:
            break
        time.sleep(0.5)
    assert attrs is not None, "Attributes should be available after processing"
    assert "gender" in attrs, "Should contain gender"
    assert "clothing_color" in attrs, "Should contain clothing_color"
    assert "apparel" in attrs, "Should contain apparel"
    ar.shutdown()

def test_attr_recognizer_different_tracks():
    from app.vision.safety_analytics import AttributeRecognizer
    ar = AttributeRecognizer(max_workers=2)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    location = (10, 160, 55, 40)
    ar.maybe_extract("track-A", frame, location)
    ar.maybe_extract("track-B", frame, location)
    # DeepFace first-load can be slow (TF init + model download)
    # Retry for up to 15 seconds
    a, b = None, None
    for _ in range(30):
        a = ar.get_attributes("track-A")
        b = ar.get_attributes("track-B")
        if a is not None and b is not None:
            break
        time.sleep(0.5)
    assert a is not None and b is not None, "Both tracks should have attributes"
    ar.shutdown()

test("Run once per track ID", test_attr_recognizer_once_per_track)
test("Handle multiple different tracks", test_attr_recognizer_different_tracks)


# ── 4. CROWD / GATHERING DETECTION ───────────────────────────────────────────
print("\n👥 4. Crowd / Gathering Detection")

def test_crowd_no_alert_few_people():
    from app.vision.safety_analytics import CrowdDetector
    alerts = []
    cd = CrowdDetector(
        proximity_px=150, min_persons=3,
        sustain_seconds=0.5, on_alert=lambda a: alerts.append(a),
    )
    # Only 2 people — shouldn't trigger
    cd.update({"p1": (100, 100), "p2": (110, 110)}, "cam-1")
    time.sleep(0.7)
    cd.update({"p1": (100, 100), "p2": (110, 110)}, "cam-1")
    assert len(alerts) == 0, f"Expected 0 alerts, got {len(alerts)}"

def test_crowd_alert_triggered():
    from app.vision.safety_analytics import CrowdDetector
    alerts = []
    cd = CrowdDetector(
        proximity_px=200, min_persons=3,
        sustain_seconds=0.3,  # Short for testing
        on_alert=lambda a: alerts.append(a),
    )
    cluster = {"p1": (100, 100), "p2": (120, 110), "p3": (110, 130)}
    cd.update(cluster, "cam-1")
    time.sleep(0.5)
    cd.update(cluster, "cam-1")
    assert len(alerts) == 1, f"Expected 1 alert, got {len(alerts)}"
    assert alerts[0]["subtype"] == "gathering"
    assert alerts[0]["person_count"] == 3

def test_crowd_no_alert_far_apart():
    from app.vision.safety_analytics import CrowdDetector
    alerts = []
    cd = CrowdDetector(
        proximity_px=50, min_persons=3,
        sustain_seconds=0.2, on_alert=lambda a: alerts.append(a),
    )
    # People are 500px apart — shouldn't cluster
    cd.update({"p1": (100, 100), "p2": (600, 600), "p3": (100, 600)}, "cam-1")
    time.sleep(0.4)
    cd.update({"p1": (100, 100), "p2": (600, 600), "p3": (100, 600)}, "cam-1")
    assert len(alerts) == 0, f"Expected 0 alerts, got {len(alerts)}"

test("No alert with < 3 people", test_crowd_no_alert_few_people)
test("Alert triggered with 3 close people", test_crowd_alert_triggered)
test("No alert when people far apart", test_crowd_no_alert_far_apart)


# ── 5. LOITERING DETECTION ───────────────────────────────────────────────────
print("\n🕐 5. Loitering / Idle Detection")

def test_loiter_no_alert_short_time():
    from app.vision.safety_analytics import LoiteringDetector
    alerts = []
    ld = LoiteringDetector(
        movement_threshold_px=50, time_window_sec=2.0,
        on_alert=lambda a: alerts.append(a),
    )
    # Feed a few data points but not enough time
    for i in range(5):
        ld.update({"p1": (100, 100)}, {"p1": "John"}, "cam-1")
        time.sleep(0.1)
    assert len(alerts) == 0, f"Expected 0 alerts (too early), got {len(alerts)}"

def test_loiter_alert_simulated():
    """Simulate loitering by manually injecting centroid history."""
    from app.vision.safety_analytics import LoiteringDetector, LoiteringTrack
    from collections import deque
    alerts = []
    # Use a 10-second window for fast testing
    ld = LoiteringDetector(
        movement_threshold_px=50, time_window_sec=10.0,
        on_alert=lambda a: alerts.append(a),
    )
    # Manually create a track with history of stationary position
    now = time.time()
    track = LoiteringTrack(person_id="p1", person_name="John")
    # Simulate 12 seconds of staying at ~(100, 100) — within the 10s window
    for i in range(40):
        t = now - 12 + (i * 0.3)  # 40 points over 12 seconds
        track.history.append((t, 100 + (i % 3), 100 + (i % 2)))

    ld._tracks["p1"] = track
    ld._check_loitering(track, "cam-1", now)
    assert len(alerts) == 1, f"Expected 1 loitering alert, got {len(alerts)}"
    assert alerts[0]["subtype"] == "loitering"
    assert alerts[0]["person_name"] == "John"

test("No alert with short observation", test_loiter_no_alert_short_time)
test("Alert with simulated stationary history", test_loiter_alert_simulated)


# ── 6. HAZARD DETECTOR ───────────────────────────────────────────────────────
print("\n🔫 6. Hazard Detector (YOLOv8)")

def test_hazard_init():
    from app.vision.hazard_detector import HazardDetector
    hd = HazardDetector(frame_interval=10, on_alert=lambda a: None)
    assert hd.frame_interval == 10
    assert hd._model is None  # Lazy loading
    hd.shutdown()

def test_hazard_frame_throttle():
    from app.vision.hazard_detector import HazardDetector
    hd = HazardDetector(frame_interval=5, on_alert=lambda a: None)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    submitted_count = 0
    for _ in range(20):
        if hd.submit_frame(frame, "cam-1"):
            submitted_count += 1
    # Every 5th frame → 20/5 = 4
    assert submitted_count == 4, f"Expected 4 submissions, got {submitted_count}"
    hd.shutdown()

def test_hazard_ultralytics_check():
    try:
        import ultralytics
        print(f"       Ultralytics v{ultralytics.__version__} installed")
    except ImportError:
        print(f"   {WARN}  ultralytics not installed — hazard detection will be disabled at runtime")

test("HazardDetector instantiation", test_hazard_init)
test("Frame throttling (every Nth)", test_hazard_frame_throttle)
test_hazard_ultralytics_check()


# ── 7. TRACKER CENTROID SUPPORT ───────────────────────────────────────────────
print("\n📍 7. Tracker Centroid Extensions")

def test_tracker_centroid_stored():
    from app.vision.tracker import PersonTracker
    tracker = PersonTracker(entry_threshold=0, exit_threshold=30)
    tracker.on_detection(
        person_id="p1", person_name="Alice", camera_id="cam-1",
        confidence=0.9, centroid=(320, 240),
    )
    centroids = tracker.get_person_centroids()
    assert "p1" in centroids, "Person should have centroid"
    assert centroids["p1"] == (320, 240), f"Wrong centroid: {centroids['p1']}"

def test_tracker_centroid_history():
    from app.vision.tracker import PersonTracker
    tracker = PersonTracker(entry_threshold=0, exit_threshold=30)
    for i in range(5):
        tracker.on_detection(
            person_id="p1", person_name="Alice", camera_id="cam-1",
            confidence=0.9, centroid=(100 + i * 10, 200),
        )
    centroids = tracker.get_person_centroids()
    assert centroids["p1"] == (140, 200), "Should have latest centroid"
    names = tracker.get_person_names()
    assert names["p1"] == "Alice"

def test_tracker_no_centroid():
    from app.vision.tracker import PersonTracker
    tracker = PersonTracker(entry_threshold=0, exit_threshold=30)
    tracker.on_detection(
        person_id="p2", person_name="Bob", camera_id="cam-1",
        confidence=0.9,  # No centroid passed
    )
    centroids = tracker.get_person_centroids()
    assert "p2" not in centroids, "Should not appear without centroid"

test("Centroid stored on detection", test_tracker_centroid_stored)
test("Centroid history updates correctly", test_tracker_centroid_history)
test("No centroid when none provided", test_tracker_no_centroid)


# ── 8. PIPELINE INTEGRATION ──────────────────────────────────────────────────
print("\n🔗 8. Pipeline Integration")

def test_pipeline_has_safety_modules():
    from app.vision.pipeline import VisionPipeline
    # Just instantiate — don't start (no camera needed)
    pipeline = VisionPipeline(camera_id="test-cam", stream_source="0")
    from app.config import settings

    if settings.ENABLE_ATTRIBUTE_RECOGNITION:
        assert pipeline.attribute_recognizer is not None, "AttributeRecognizer should be set"
    if settings.ENABLE_CROWD_DETECTION:
        assert pipeline.crowd_detector is not None, "CrowdDetector should be set"
    if settings.ENABLE_LOITERING_DETECTION:
        assert pipeline.loitering_detector is not None, "LoiteringDetector should be set"
    if settings.ENABLE_HAZARD_DETECTION:
        assert pipeline.hazard_detector is not None, "HazardDetector should be set"

def test_pipeline_status_includes_safety():
    from app.vision.pipeline import VisionPipeline
    pipeline = VisionPipeline(camera_id="test-cam", stream_source="0")
    status = pipeline.status
    assert "safety_modules" in status, "Status should include safety_modules"
    sm = status["safety_modules"]
    assert "attribute_recognition" in sm
    assert "crowd_detection" in sm
    assert "loitering_detection" in sm
    assert "hazard_detection" in sm

def test_pipeline_security_alert_callback():
    from app.vision.pipeline import VisionPipeline
    events = []
    pipeline = VisionPipeline(
        camera_id="test-cam", stream_source="0",
        on_event=lambda e: events.append(e),
    )
    # Simulate a security alert
    pipeline._handle_security_alert({
        "event_type": "security_alert",
        "subtype": "gathering",
        "camera_id": "test-cam",
    })
    assert len(events) == 1
    assert events[0]["event_type"] == "security_alert"

test("Pipeline instantiates safety modules", test_pipeline_has_safety_modules)
test("Pipeline status includes safety info", test_pipeline_status_includes_safety)
test("Security alerts route through on_event", test_pipeline_security_alert_callback)


# ── 9. DEPENDENCY CHECK ──────────────────────────────────────────────────────
print("\n📋 9. Dependency Status")

deps = {
    "deepface": "Gender recognition",
    "ultralytics": "Hazard detection (YOLOv8)",
    "scipy": "Distance computations",
    "cv2": "Image processing",
    "numpy": "Numerical operations",
}

for pkg, purpose in deps.items():
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "unknown")
        print(f"  {PASS}  {pkg} v{ver}  ({purpose})")
    except ImportError:
        print(f"  {WARN}  {pkg} NOT INSTALLED  ({purpose})")


# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
passed = sum(results)
total = len(results)
if passed == total:
    print(f"  🎉 ALL {total} TESTS PASSED!")
else:
    print(f"  ⚠  {passed}/{total} tests passed, {total - passed} failed")
print("=" * 60 + "\n")

sys.exit(0 if passed == total else 1)
