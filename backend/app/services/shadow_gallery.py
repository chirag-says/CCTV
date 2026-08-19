"""
Shadow Gallery — Background gallery validation system.

Maintains a fixed validation set of representative images and periodically
evaluates the gallery's recognition accuracy against them. Runs completely
in the background — never in the realtime pipeline.

Architecture:
  ┌─────────────────────┐
  │  Main Gallery       │ ── realtime recognition
  └─────────────────────┘
           │
           ▼  (background, async)
  ┌─────────────────────┐
  │  Shadow Gallery     │ ── periodic validation
  │  (validation set)   │
  └─────────────────────┘
           │
           ▼
  ┌─────────────────────┐
  │  Gallery Score      │ ── tracks accuracy over time
  │  History            │
  └─────────────────────┘

Usage:
  - Validation images are stored in `snapshots/validation/`
  - Each person has 1-3 known test images NOT in the gallery
  - The shadow gallery runs every N minutes and reports:
    * True positive rate (correct identification)
    * False positive rate (wrong identification)
    * Miss rate (failed to identify)
    * Per-person accuracy breakdown
"""

import os
import json
import time
import threading
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

import numpy as np
import cv2

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a single validation image test."""
    person_id: str
    person_name: str
    expected: bool  # True if this person should be in the gallery
    matched: bool   # True if the gallery matched them
    matched_as: Optional[str] = None  # Who the gallery thought they were
    distance: float = 0.0
    correct: bool = False


@dataclass
class GalleryScore:
    """Aggregate score for a gallery validation run."""
    timestamp: float
    gallery_version: int
    total_tests: int
    true_positives: int  # Correctly identified known person
    false_positives: int  # Identified as wrong person
    misses: int  # Failed to identify known person
    accuracy: float  # TP / total_tests
    per_person: Dict[str, dict] = field(default_factory=dict)
    run_duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "gallery_version": self.gallery_version,
            "total_tests": self.total_tests,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "misses": self.misses,
            "accuracy": round(self.accuracy * 100, 1),
            "per_person": self.per_person,
            "run_duration_ms": round(self.run_duration_ms, 1),
        }


class ShadowGallery:
    """
    Background gallery validation engine.

    Validation set structure:
      snapshots/validation/
        ├── manifest.json      # {person_id: {name, images: [path, ...]}}
        ├── person_abc/
        │   ├── test_1.jpg
        │   └── test_2.jpg
        └── person_def/
            └── test_1.jpg

    The manifest maps person_ids to their validation images. Only persons
    in the manifest are tested — the rest of the gallery is untouched.
    """

    VALIDATION_DIR = os.path.join("snapshots", "validation")
    MANIFEST_FILE = os.path.join("snapshots", "validation", "manifest.json")
    HISTORY_FILE = os.path.join("logs", "shadow_gallery_history.json")

    def __init__(
        self,
        interval_minutes: int = 30,
        max_history: int = 100,
    ):
        self._interval = interval_minutes * 60
        self._max_history = max_history
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

        # Score history (most recent last)
        self._history: List[GalleryScore] = []
        self._load_history()

        # Last known gallery version to detect changes
        self._last_gallery_version = -1

    def start(self):
        """Start the background validation thread."""
        if self._running:
            return
        if not os.path.exists(self.MANIFEST_FILE):
            logger.info(
                f"⚠️ Shadow Gallery: no manifest at {self.MANIFEST_FILE}, "
                f"skipping background validation. Run /api/system-health/shadow-gallery/init "
                f"to create the validation set."
            )
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, name="ShadowGallery", daemon=True
        )
        self._thread.start()
        self._running = True
        logger.info(f"🔮 Shadow Gallery started (interval={self._interval}s)")

    def stop(self):
        """Stop the background thread."""
        if not self._running:
            return
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10.0)
        self._running = False
        logger.info("🔮 Shadow Gallery stopped")

    def _run_loop(self):
        """Main loop: run validation at configured interval."""
        # Initial delay to let the system stabilize
        time.sleep(30)

        while not self._stop_event.is_set():
            try:
                # Check if gallery version changed
                current_version = self._get_gallery_version()
                if current_version != self._last_gallery_version:
                    logger.info(
                        f"🔮 Gallery version changed: {self._last_gallery_version} → "
                        f"{current_version}, running validation..."
                    )
                    score = self.run_validation(current_version)
                    if score:
                        self._history.append(score)
                        if len(self._history) > self._max_history:
                            self._history = self._history[-self._max_history:]
                        self._save_history()
                        self._last_gallery_version = current_version

                        logger.info(
                            f"🔮 Shadow Gallery validation: "
                            f"accuracy={score.accuracy * 100:.1f}% "
                            f"({score.true_positives}/{score.total_tests} correct, "
                            f"{score.false_positives} FP, {score.misses} misses) "
                            f"in {score.run_duration_ms:.0f}ms"
                        )

                        # Alert if accuracy drops below threshold
                        if score.accuracy < 0.8 and score.total_tests >= 3:
                            logger.warning(
                                f"⚠️ SHADOW GALLERY ALERT: accuracy dropped to "
                                f"{score.accuracy * 100:.1f}% — investigate gallery quality!"
                            )
            except Exception as e:
                logger.error(f"Shadow Gallery error: {e}", exc_info=True)

            # Sleep in small increments for responsive shutdown
            for _ in range(int(self._interval / 5)):
                if self._stop_event.is_set():
                    return
                time.sleep(5)

    def run_validation(self, gallery_version: int = 0) -> Optional[GalleryScore]:
        """
        Run a validation pass against the current gallery.

        Creates a lightweight detector+recognizer pair loaded with current
        gallery data. Runs entirely in background — no pipeline impact.

        Returns:
            GalleryScore with accuracy metrics, or None if no validation data.
        """
        manifest = self._load_manifest()
        if not manifest:
            return None

        start = time.perf_counter()
        results: List[ValidationResult] = []

        try:
            # Create standalone detector + recognizer for validation
            from app.vision.detector import FaceDetector
            from app.vision.recognizer import FaceRecognizer
            from app.services.person_service import PersonService

            detector = FaceDetector()
            recognizer = FaceRecognizer()

            # Load current gallery encodings
            encodings = PersonService.get_all_encodings()
            if encodings:
                recognizer.load_encodings(encodings)
            else:
                logger.warning("Shadow Gallery: no encodings to validate against")
                return None

            for person_id, person_data in manifest.items():
                person_name = person_data.get("name", "Unknown")
                images = person_data.get("images", [])

                for image_path in images:
                    full_path = os.path.join(self.VALIDATION_DIR, image_path)
                    if not os.path.exists(full_path):
                        logger.debug(f"Shadow Gallery: missing image {full_path}")
                        continue

                    result = self._test_image(
                        detector, recognizer, full_path, person_id, person_name
                    )
                    if result:
                        results.append(result)

        except Exception as e:
            logger.error(f"Shadow Gallery validation failed: {e}", exc_info=True)
            return None

        if not results:
            return None

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Compute aggregate score
        tp = sum(1 for r in results if r.correct and r.matched)
        fp = sum(1 for r in results if r.matched and not r.correct)
        misses = sum(1 for r in results if not r.matched and r.expected)

        # Per-person breakdown
        per_person: Dict[str, dict] = {}
        for r in results:
            if r.person_id not in per_person:
                per_person[r.person_id] = {
                    "name": r.person_name,
                    "tests": 0, "correct": 0,
                    "wrong": 0, "missed": 0,
                }
            pp = per_person[r.person_id]
            pp["tests"] += 1
            if r.correct:
                pp["correct"] += 1
            elif r.matched:
                pp["wrong"] += 1
            else:
                pp["missed"] += 1

        return GalleryScore(
            timestamp=time.time(),
            gallery_version=gallery_version,
            total_tests=len(results),
            true_positives=tp,
            false_positives=fp,
            misses=misses,
            accuracy=tp / len(results) if results else 0,
            per_person=per_person,
            run_duration_ms=elapsed_ms,
        )

    def _test_image(
        self,
        detector,
        recognizer,
        image_path: str,
        expected_person_id: str,
        expected_person_name: str,
    ) -> Optional[ValidationResult]:
        """Test a single validation image against the gallery."""
        try:
            # Read the image (BGR, as detector expects)
            img = cv2.imread(image_path)
            if img is None:
                return None

            # Detect faces and extract embeddings in one pass
            locations, embeddings, _ = detector.detect_faces_with_embeddings(img)

            if not embeddings or all(e is None for e in embeddings):
                return ValidationResult(
                    person_id=expected_person_id,
                    person_name=expected_person_name,
                    expected=True,
                    matched=False,
                    correct=False,
                )

            # Use the first valid embedding
            embedding = None
            for emb in embeddings:
                if emb is not None:
                    embedding = emb
                    break

            if embedding is None:
                return ValidationResult(
                    person_id=expected_person_id,
                    person_name=expected_person_name,
                    expected=True,
                    matched=False,
                    correct=False,
                )

            # Run recognition
            result = recognizer.recognize_face(embedding)

            if result and result.matched and result.person_id:
                correct = result.person_id == expected_person_id
                return ValidationResult(
                    person_id=expected_person_id,
                    person_name=expected_person_name,
                    expected=True,
                    matched=True,
                    matched_as=result.person_name,
                    distance=result.distance,
                    correct=correct,
                )
            else:
                return ValidationResult(
                    person_id=expected_person_id,
                    person_name=expected_person_name,
                    expected=True,
                    matched=False,
                    distance=result.distance if result else 0,
                    correct=False,
                )

        except Exception as e:
            logger.debug(f"Shadow Gallery: test failed for {image_path}: {e}")
            return None

    def _load_manifest(self) -> dict:
        """Load the validation manifest."""
        try:
            if not os.path.exists(self.MANIFEST_FILE):
                return {}
            with open(self.MANIFEST_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load shadow gallery manifest: {e}")
            return {}

    def _load_history(self):
        """Load score history from disk."""
        try:
            if os.path.exists(self.HISTORY_FILE):
                with open(self.HISTORY_FILE, 'r') as f:
                    raw = json.load(f)
                self._history = [
                    GalleryScore(
                        timestamp=s["timestamp"],
                        gallery_version=s.get("gallery_version", 0),
                        total_tests=s["total_tests"],
                        true_positives=s["true_positives"],
                        false_positives=s["false_positives"],
                        misses=s["misses"],
                        accuracy=s["accuracy"] / 100,  # stored as percentage
                        per_person=s.get("per_person", {}),
                        run_duration_ms=s.get("run_duration_ms", 0),
                    )
                    for s in raw
                ]
        except Exception as e:
            logger.debug(f"Failed to load shadow gallery history: {e}")
            self._history = []

    def _save_history(self):
        """Persist score history to disk."""
        try:
            os.makedirs(os.path.dirname(self.HISTORY_FILE) or ".", exist_ok=True)
            with open(self.HISTORY_FILE, 'w') as f:
                json.dump([s.to_dict() for s in self._history], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save shadow gallery history: {e}")

    @staticmethod
    def _get_gallery_version() -> int:
        """Get current gallery version from DB."""
        try:
            from app.db.session import SessionLocal
            from app.db.models import EncodingVersion
            db = SessionLocal()
            try:
                row = db.query(EncodingVersion).filter(EncodingVersion.id == 1).first()
                return row.version if row else 0
            finally:
                db.close()
        except Exception:
            return 0

    @staticmethod
    def create_validation_set() -> dict:
        """
        Auto-generate a validation set from the current gallery.

        Strategy: for each person with 3+ encodings, hold out the
        most recent encoding's source image as a validation image
        (if it exists on disk). This ensures the validation set
        contains images the gallery has never seen.

        Returns:
            {person_id: {name, images}} manifest dict
        """
        try:
            from app.db.session import SessionLocal
            from app.db.models import Person, FaceEncoding

            db = SessionLocal()
            try:
                manifest = {}
                validation_dir = ShadowGallery.VALIDATION_DIR
                os.makedirs(validation_dir, exist_ok=True)

                persons = db.query(Person).filter(Person.is_active == True).all()

                for person in persons:
                    encodings = (
                        db.query(FaceEncoding)
                        .filter(FaceEncoding.person_id == person.id)
                        .order_by(FaceEncoding.created_at.desc())
                        .all()
                    )

                    if len(encodings) < 3:
                        continue  # Not enough encodings to hold one out

                    # Find the most recent encoding with a source image on disk
                    validation_images = []
                    for enc in encodings[:2]:  # Check the 2 most recent
                        src = enc.source_image
                        if not src:
                            continue

                        # Resolve path: DB stores URL paths like /snapshots/...
                        # On disk they live at relative paths snapshots/...
                        disk_path = src.lstrip("/")
                        if not os.path.exists(disk_path):
                            # Try absolute as-is (Linux)
                            if not os.path.exists(src):
                                continue
                            disk_path = src

                        # Copy to validation directory
                        person_dir = os.path.join(validation_dir, person.id[:12])
                        os.makedirs(person_dir, exist_ok=True)
                        dest = os.path.join(person_dir, f"{enc.id[:8]}.jpg")

                        if not os.path.exists(dest):
                            import shutil
                            shutil.copy2(disk_path, dest)

                        # Relative path from validation dir
                        rel_path = os.path.join(person.id[:12], f"{enc.id[:8]}.jpg")
                        validation_images.append(rel_path)

                    # Fallback: try avatar_url if no encoding source images found
                    if not validation_images and person.avatar_url:
                        avatar_disk = person.avatar_url.lstrip("/")
                        if os.path.exists(avatar_disk):
                            person_dir = os.path.join(validation_dir, person.id[:12])
                            os.makedirs(person_dir, exist_ok=True)
                            dest = os.path.join(person_dir, "avatar.jpg")
                            if not os.path.exists(dest):
                                import shutil
                                shutil.copy2(avatar_disk, dest)
                            validation_images.append(os.path.join(person.id[:12], "avatar.jpg"))

                    if validation_images:
                        manifest[person.id] = {
                            "name": person.name,
                            "images": validation_images,
                        }

                # Save manifest
                manifest_path = ShadowGallery.MANIFEST_FILE
                with open(manifest_path, 'w') as f:
                    json.dump(manifest, f, indent=2)

                logger.info(
                    f"🔮 Shadow Gallery validation set created: "
                    f"{len(manifest)} persons, "
                    f"{sum(len(p['images']) for p in manifest.values())} images"
                )

                return manifest

            finally:
                db.close()
        except Exception as e:
            logger.error(f"Failed to create validation set: {e}", exc_info=True)
            return {}

    @property
    def stats(self) -> dict:
        """Return shadow gallery stats for the dashboard."""
        latest = self._history[-1] if self._history else None
        trend = None
        if len(self._history) >= 2:
            prev = self._history[-2]
            trend = round((latest.accuracy - prev.accuracy) * 100, 1) if latest else None

        return {
            "running": self._running,
            "interval_minutes": self._interval // 60,
            "total_validations": len(self._history),
            "latest": latest.to_dict() if latest else None,
            "accuracy_trend": trend,
            "has_manifest": os.path.exists(self.MANIFEST_FILE),
        }


# ── Singleton ─────────────────────────────────────────────────────

_shadow_gallery: Optional[ShadowGallery] = None
_shadow_lock = threading.Lock()


def get_shadow_gallery() -> ShadowGallery:
    """Get or create the singleton ShadowGallery."""
    global _shadow_gallery
    if _shadow_gallery is None:
        with _shadow_lock:
            if _shadow_gallery is None:
                try:
                    from app.config import settings
                    interval = getattr(settings, 'SHADOW_GALLERY_INTERVAL_MIN', 30)
                except Exception:
                    interval = 30
                _shadow_gallery = ShadowGallery(interval_minutes=interval)
    return _shadow_gallery
