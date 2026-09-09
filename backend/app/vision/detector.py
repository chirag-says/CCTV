"""
Face Detection Module — SCRFD via InsightFace.

Detects faces using the SCRFD model from insightface, which provides:
- High-accuracy face bounding boxes
- 5-point facial landmarks (for alignment before ArcFace embedding)
- GPU acceleration via ONNX Runtime

Replaces the old dlib/face_recognition-based detector.
"""

import os
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from app.config import settings

logger = logging.getLogger(__name__)

# Lazy-loaded insightface app (shared across instances for VRAM efficiency)
_insightface_app = None
_insightface_lock = None
_insightface_failed = False


# DLL directories registered for onnxruntime's CUDA provider. Handles must be
# kept alive: os.add_dll_directory() removes the directory again when the
# returned object is closed or garbage collected.
_cuda_dll_handles = []


def _ensure_cuda_dll_path() -> list:
    """
    Make onnxruntime able to FIND the CUDA runtime, explicitly.

    THE BUG THIS FIXES
    onnxruntime-gpu does not ship the CUDA runtime. On this machine the only
    copy of `cublasLt64_12.dll` lives in site-packages/torch/lib -- PyTorch
    bundles its own. onnxruntime does not look there, so creating a CUDA
    session fails with:

        Error loading "onnxruntime_providers_cuda.dll" which depends on
        "cublasLt64_12.dll" which is missing. (Error 126)

    and silently falls back to CPU. Measured cost of that fallback: SCRFD at
    det=1024 goes from 34 ms to 3,648 ms. A ~100x regression, on a code path
    whose only symptom is "the cameras got slow".

    It USUALLY worked anyway, for a bad reason: importing torch loads those
    DLLs into the process, so if the hazard detector or ReID happened to
    initialise before the first frame reached insightface, onnxruntime could
    resolve them. Every `import torch` in this codebase is inside a function,
    so that ordering was a race decided at startup -- and losing it cost 100x.

    This makes the dependency explicit, so the outcome no longer depends on
    which subsystem happened to wake up first.

    Returns a description of what was registered, for logging.
    """
    added = []

    # 1. THE ONE THAT ACTUALLY WORKS: import torch.
    #
    # Registering torch/lib with os.add_dll_directory() is NOT sufficient --
    # measured: onnxruntime still resolved to CPUExecutionProvider. It loads
    # its provider DLL with search flags that bypass user-added directories.
    #
    # Importing torch loads the CUDA runtime into the process by absolute
    # path, after which Windows resolves onnxruntime's dependency on
    # `cublasLt64_12.dll` against the already-loaded module. This is precisely
    # what used to happen by luck, whenever some other subsystem imported
    # torch before the first frame reached insightface. Doing it here makes it
    # a guarantee instead of a race.
    #
    # torch is already a hard dependency (YOLO, ReID) and is imported anyway,
    # so this moves the cost rather than adding it.
    try:
        import torch  # noqa: F401
        added.append("torch runtime")
    except Exception as e:
        logger.warning(
            f"Could not import torch to provide the CUDA runtime: {e}. "
            f"onnxruntime may fall back to CPU (~100x slower)."
        )

    # 2. Belt and braces for the `nvidia-*-cu12` wheel layout, where the
    #    runtime is installed properly rather than borrowed from torch.
    if hasattr(os, "add_dll_directory") and not _cuda_dll_handles:
        import importlib.util
        candidates = []
        for pkg, sub in (("torch", "lib"), ("nvidia", None)):
            try:
                spec = importlib.util.find_spec(pkg)
                if not (spec and spec.submodule_search_locations):
                    continue
                root = Path(list(spec.submodule_search_locations)[0])
                candidates.extend([root / sub] if sub else sorted(root.glob("*/bin")))
            except Exception:
                pass
        for d in candidates:
            try:
                if d.is_dir():
                    _cuda_dll_handles.append(os.add_dll_directory(str(d)))
                    added.append(str(d))
            except Exception as e:
                logger.debug(f"Could not add DLL directory {d}: {e}")

    return added


class CPUInferenceRefused(RuntimeError):
    """Inference would run on the CPU and settings.ALLOW_CPU_INFERENCE is off.

    Deliberately fatal. See settings.ALLOW_CPU_INFERENCE for the measured
    cost of letting this through instead.
    """


def _conflicting_onnxruntime():
    """Detect the CPU onnxruntime wheel installed beside the GPU one.

    They ship the SAME `onnxruntime` package. Whichever wrote its files last
    wins the import, so a stray `pip install onnxruntime` silently replaces
    the CUDA-capable build with one that has no CUDA provider at all -- and
    nothing about the resulting behaviour says so.
    """
    try:
        from importlib.metadata import distributions
        found = sorted({
            (d.metadata["Name"] or "") for d in distributions()
            if (d.metadata["Name"] or "").lower().startswith("onnxruntime")
        })
        if len(found) > 1 and any(f.lower() == "onnxruntime" for f in found):
            return ", ".join(found)
    except Exception:
        pass
    return None


def _cpu_fallback_message() -> str:
    """Say which of the handful of causes this actually is."""
    try:
        import onnxruntime as ort
        avail, ver = ort.get_available_providers(), ort.__version__
    except Exception:
        avail, ver = ["<onnxruntime import failed>"], "?"

    bar = "=" * 70
    lines = [
        "",
        bar,
        "  InsightFace has no CUDA execution provider, so SCRFD and ArcFace",
        "  would run on the CPU. Measured on this project's hardware that is",
        "  3,648 ms per detection instead of 34 ms -- a 100x regression whose",
        "  only symptom is that the cameras get slow.",
        "",
        "  onnxruntime {} offers: {}".format(ver, avail),
    ]
    clash = _conflicting_onnxruntime()
    if clash:
        lines += [
            "",
            "  LIKELY CAUSE: {} are both installed. They are the".format(clash),
            "  same package; the CPU wheel wins the import and has no CUDA.",
            "    pip uninstall -y onnxruntime",
            "    pip install --force-reinstall 'onnxruntime-gpu>=1.20'",
        ]
    else:
        lines += [
            "",
            "  Blackwell (sm_120) needs onnxruntime-gpu >= 1.20:",
            "    pip install -U 'onnxruntime-gpu>=1.20'",
            "  Never install plain `onnxruntime` alongside it.",
        ]
    lines += [
        "",
        "  To run on the CPU deliberately (an API host with no cameras, CI):",
        "    ALLOW_CPU_INFERENCE=true",
        bar,
    ]
    return "\n".join(lines)


def _get_insightface_app():
    """
    Lazy-load the insightface FaceAnalysis app.

    The app bundles both SCRFD (detection) and ArcFace (recognition),
    but we only use the detection model here. The recognition model
    is used separately in recognizer.py.

    We keep a module-level singleton to avoid loading the model
    multiple times (each load consumes GPU VRAM).
    """
    global _insightface_app, _insightface_lock, _insightface_failed
    import threading

    if _insightface_lock is None:
        _insightface_lock = threading.Lock()

    with _insightface_lock:
        if _insightface_app is not None:
            return _insightface_app

        if _insightface_failed:
            return None

        try:
            # MUST run before onnxruntime creates a CUDA session: DLL search
            # paths only affect libraries loaded after they are registered.
            added = _ensure_cuda_dll_path()
            if added:
                logger.info(f"CUDA DLL search path: {', '.join(added)}")

            import insightface
            from insightface.app import FaceAnalysis

            model_name = settings.INSIGHTFACE_MODEL_NAME
            det_size = settings.INSIGHTFACE_DET_SIZE

            logger.info(f"Loading insightface model: {model_name} (det_size={det_size})")

            # Request CUDA explicitly — do NOT silently fall back to CPU.
            # ONNX Runtime 1.17 predates Blackwell (sm_120) and will silently
            # use CPUExecutionProvider if CUDA fails, turning 10 ms inference
            # into 200-500 ms. Fail loudly instead.
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            # Only the modules whose output is actually read. See
            # settings.INSIGHTFACE_MODULES for what the others cost.
            modules_cfg = getattr(settings, "INSIGHTFACE_MODULES", "") or ""
            allowed = [m.strip() for m in modules_cfg.split(",") if m.strip()]
            app = FaceAnalysis(
                name=model_name,
                providers=providers,
                allowed_modules=allowed or None,
            )
            app.prepare(
                ctx_id=0,  # GPU 0
                det_size=(det_size, det_size),
                det_thresh=settings.INSIGHTFACE_DET_THRESH,
            )

            # Verify CUDA is actually being used, not just listed
            _cuda_active = False
            for model_name_key in ['detection', 'recognition']:
                model = app.models.get(model_name_key)
                if model and hasattr(model, 'session'):
                    active_providers = model.session.get_providers()
                    logger.info(
                        f"InsightFace {model_name_key} providers: {active_providers}"
                    )
                    if 'CUDAExecutionProvider' in active_providers:
                        _cuda_active = True

            if not _cuda_active:
                if not getattr(settings, "ALLOW_CPU_INFERENCE", False):
                    raise CPUInferenceRefused(_cpu_fallback_message())
                logger.error(
                    "InsightFace is on the CPU and ALLOW_CPU_INFERENCE is set "
                    "-- continuing at roughly 1/100th speed."
                    + _cpu_fallback_message()
                )

            _insightface_app = app
            logger.info(
                f"InsightFace loaded successfully: {model_name}, "
                f"det_size={det_size}, thresh={settings.INSIGHTFACE_DET_THRESH}, "
                f"modules={sorted(app.models.keys())}, "
                f"CUDA={'active' if _cuda_active else 'INACTIVE (CPU fallback!)'}"
            )
            return _insightface_app

        except CPUInferenceRefused:
            # Deliberate, and must not be swallowed into a None return: the
            # caller reads None as "no faces this frame", so a refusal that
            # got caught here would present as a camera that sees nobody.
            _insightface_failed = True
            raise
        except Exception as e:
            logger.error(f"Failed to load insightface: {e}", exc_info=True)
            _insightface_failed = True
            return None


class FaceDetector:
    """
    Detects faces in video frames using SCRFD (via insightface).

    Features:
    - GPU-accelerated face detection (ONNX Runtime + CUDA)
    - 5-point facial landmark output (for face alignment)
    - Bounding box extraction
    - Face quality assessment
    - Backward-compatible (top, right, bottom, left) output format
    """

    # Minimum face size for detection (tracking + counting).
    # Separate from recognition gate — small faces are still useful
    # for tracking, entry/exit events, and crowd counting.
    MIN_FACE_SIZE_DEFAULT = 32

    def __init__(
        self,
        min_face_size: int = None,
        scale_factor: float = None,
        **kwargs,  # Accept extra kwargs for backward compat
    ):
        self.min_face_size = min_face_size or self.MIN_FACE_SIZE_DEFAULT
        self.scale_factor = scale_factor or settings.DETECTION_SCALE
        self._frame_count = 0
        self._app = None  # Lazy-loaded

        logger.info(
            f"FaceDetector initialized (SCRFD): "
            f"model={settings.INSIGHTFACE_MODEL_NAME}, "
            f"min_face={self.min_face_size}"
        )

    def _get_app(self):
        """Get the shared insightface app (lazy-loaded on first use)."""
        if self._app is None:
            self._app = _get_insightface_app()
        return self._app

    @staticmethod
    def _is_valid_face_geometry(face) -> bool:
        """
        Validate that the 5-point facial landmarks form a plausible face.

        InsightFace's SCRFD provides 5 landmarks:
          [0] left eye, [1] right eye, [2] nose tip,
          [3] left mouth corner, [4] right mouth corner

        Vehicle headlights, reflections, and license plates can trigger
        false positives but their "landmark" positions are geometrically
        implausible as faces. This method checks:
          1. Eyes should be roughly horizontal (not vertically stacked)
          2. Nose should be between the eyes (horizontally) and below them
          3. Mouth should be below the nose
          4. The face should have a reasonable height-to-width ratio
          5. Detection confidence should be above a secondary threshold

        Returns True if the detection looks like a real face.
        """
        # Check detection score (secondary threshold above SCRFD's base)
        det_score = getattr(face, 'det_score', None)
        if det_score is not None and det_score < 0.45:
            return False

        # Get landmarks (5-point)
        kps = getattr(face, 'kps', None)
        if kps is None or len(kps) < 5:
            # No landmarks available — can't validate, allow it through
            return True

        try:
            left_eye = kps[0]    # (x, y)
            right_eye = kps[1]
            nose = kps[2]
            left_mouth = kps[3]
            right_mouth = kps[4]

            # 1. Eye separation: eyes should be spread apart horizontally
            eye_dx = abs(right_eye[0] - left_eye[0])
            eye_dy = abs(right_eye[1] - left_eye[1])
            if eye_dx < 5:
                # Eyes are nearly vertically stacked — not a face
                return False
            # Eyes should be roughly horizontal (allow up to ~40° tilt)
            if eye_dy > eye_dx * 0.85:
                return False

            # 2. Nose should be roughly between the eyes horizontally
            eye_cx = (left_eye[0] + right_eye[0]) / 2
            nose_offset = abs(nose[0] - eye_cx)
            if nose_offset > eye_dx * 1.0:
                # Nose is way outside the eye region — not a face
                return False

            # 3. Nose should be below or at eye level
            eye_cy = (left_eye[1] + right_eye[1]) / 2
            if nose[1] < eye_cy - eye_dx * 0.3:
                # Nose is significantly above the eyes — not a face
                return False

            # 4. Mouth should be below the nose
            mouth_cy = (left_mouth[1] + right_mouth[1]) / 2
            if mouth_cy < nose[1] - eye_dx * 0.2:
                # Mouth is above the nose — not a face
                return False

            # 5. Overall face proportion check
            # Face height (eyes to mouth) should be reasonable vs width (eye span)
            face_height = mouth_cy - eye_cy
            if face_height < 0:
                return False  # Mouth above eyes
            if eye_dx > 0 and face_height / eye_dx > 4.0:
                # Face is extremely tall and narrow — likely not a face
                return False

            return True

        except (IndexError, TypeError):
            # If landmark parsing fails, allow the detection through
            return True

    def detect_faces(
        self, frame: np.ndarray
    ) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
        """
        Detect faces in a frame.

        Args:
            frame: BGR image from OpenCV (numpy array)

        Returns:
            Tuple of (face_locations, frame)
            face_locations: List of (top, right, bottom, left) tuples
                            (backward-compatible format)
            frame: The original BGR frame (passed through unchanged)

        Note: InsightFace expects BGR input — it does swapRB=True
        internally in both SCRFD and ArcFace. Do NOT convert to RGB.
        """
        # Pass BGR directly — InsightFace handles BGR→RGB internally
        # via cv2.dnn.blobFromImage(swapRB=True)

        # Scale down for performance if needed
        if self.scale_factor < 1.0:
            small_frame = cv2.resize(
                frame, (0, 0),
                fx=self.scale_factor,
                fy=self.scale_factor,
            )
        else:
            small_frame = frame

        # Run SCRFD detection
        app = self._get_app()
        if app is None:
            self._frame_count += 1
            return [], frame

        faces = app.get(small_frame)

        # Extract bounding boxes and scale back
        face_locations = []
        for face in faces:
            bbox = face.bbox.astype(int)  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox

            # Scale back to original frame size
            if self.scale_factor < 1.0:
                inv_scale = 1.0 / self.scale_factor
                x1 = int(x1 * inv_scale)
                y1 = int(y1 * inv_scale)
                x2 = int(x2 * inv_scale)
                y2 = int(y2 * inv_scale)

            # Filter out too-small faces
            face_w = x2 - x1
            face_h = y2 - y1
            if face_w < self.min_face_size or face_h < self.min_face_size:
                continue

            # Validate face landmark geometry to reject non-face detections
            # (vehicles, reflections, etc. can trigger false positives)
            if not self._is_valid_face_geometry(face):
                continue

            # Convert to (top, right, bottom, left) format for backward compat
            face_locations.append((y1, x2, y2, x1))

        self._frame_count += 1
        return face_locations, frame

    def detect_faces_with_embeddings(
        self, frame: np.ndarray
    ) -> Tuple[List[Tuple[int, int, int, int]], List, List, np.ndarray]:
        """
        Detect faces AND extract ArcFace embeddings in a SINGLE pass.

        This is the most efficient method — runs insightface once on the
        full-resolution frame, returning both bounding boxes and 512-d
        embeddings. No separate crop-and-re-detect needed.

        Sub-stage timing is recorded in the profiler:
          scrfd_preprocess  — (no-op now, color conversion removed)
          scrfd_inference   — insightface app.get() (ONNX forward pass)
          scrfd_postprocess — bbox/embedding extraction + filtering

        Args:
            frame: BGR image from OpenCV (numpy array)

        Returns:
            Tuple of (face_locations, embeddings, landmarks, frame)
            face_locations: List of (top, right, bottom, left)
            embeddings: List of 512-d numpy arrays (or None per face)
            landmarks: List of SCRFD 5-point landmark arrays (or None per face).
                       Same index order as face_locations — needed downstream
                       for real yaw/pitch/roll estimation.
            frame: The original BGR frame (passed through unchanged)

        Note: InsightFace expects BGR input — it does swapRB=True
        internally in both SCRFD and ArcFace. Do NOT convert to RGB.
        """
        from app.core.profiler import profiler

        # ── Preprocess ────────────────────────────────────────────
        # No color conversion needed: InsightFace handles BGR→RGB
        # internally via cv2.dnn.blobFromImage(swapRB=True).
        # Removing the cvtColor saves ~1-2 ms and fixes the double-swap
        # bug that was degrading both detection and recognition accuracy.

        app = self._get_app()
        if app is None:
            self._frame_count += 1
            return [], [], [], frame

        # ── Inference (GPU via ONNX) ──────────────────────────────
        with profiler.measure("scrfd_inference"):
            faces = app.get(frame)

        # ── Postprocess (CPU) ─────────────────────────────────────
        with profiler.measure("scrfd_postprocess"):
            face_locations = []
            embeddings = []
            landmarks = []
            for face in faces:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox

                # Filter too-small faces
                face_w = x2 - x1
                face_h = y2 - y1
                if face_w < self.min_face_size or face_h < self.min_face_size:
                    continue

                # Validate face landmark geometry to reject non-face detections
                if not self._is_valid_face_geometry(face):
                    continue

                # Convert to (top, right, bottom, left)
                face_locations.append((y1, x2, y2, x1))

                # Extract embedding (already computed by app.get)
                if hasattr(face, 'embedding') and face.embedding is not None:
                    embeddings.append(face.embedding)
                else:
                    embeddings.append(None)

                # Carry the 5-point landmarks through. app.get() already
                # computed them for alignment; passing them on costs nothing
                # and is what lets pose estimation return a real yaw instead
                # of (0, 0, 0).
                kps = getattr(face, "kps", None)
                landmarks.append(kps if kps is not None and len(kps) >= 5 else None)

        # Record face count in profiler
        profiler.record_faces_detected(len(face_locations))

        self._frame_count += 1
        return face_locations, embeddings, landmarks, frame

    def detect_faces_raw(
        self, frame_bgr: np.ndarray
    ) -> list:
        """
        Detect faces and return raw insightface Face objects.
        These contain bbox, landmarks, and embedding if the recognition
        model is loaded.

        Args:
            frame_bgr: BGR image from OpenCV (numpy array).
                       InsightFace handles BGR→RGB internally.

        Returns:
            List of insightface Face objects
        """
        app = self._get_app()
        if app is None:
            return []
        faces = app.get(frame_bgr)
        return [f for f in faces if self._is_valid_face_geometry(f)]

    def extract_face_crop(
        self,
        frame: np.ndarray,
        location: Tuple[int, int, int, int],
        padding: int = 20,
    ) -> np.ndarray:
        """
        Extract a cropped face from a frame with padding.

        Args:
            frame: Original frame (BGR)
            location: (top, right, bottom, left) bounding box
            padding: Extra pixels around the face

        Returns:
            Cropped face image (BGR)
        """
        top, right, bottom, left = location
        h, w = frame.shape[:2]

        # Add padding, clamped to frame bounds
        top = max(0, top - padding)
        right = min(w, right + padding)
        bottom = min(h, bottom + padding)
        left = max(0, left - padding)

        return frame[top:bottom, left:right].copy()

    def assess_face_quality(
        self, frame: np.ndarray, location: Tuple[int, int, int, int]
    ) -> float:
        """
        Assess face image quality (0.0 to 1.0).

        Considers:
        - Face size (larger = better)
        - Blur level (sharper = better)
        - Brightness (balanced = better)
        """
        top, right, bottom, left = location
        face_crop = frame[top:bottom, left:right]

        if face_crop.size == 0:
            return 0.0

        # Size score (normalize face area)
        face_area = (bottom - top) * (right - left)
        size_score = min(1.0, face_area / (150 * 150))

        # Blur score (Laplacian variance)
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(1.0, laplacian_var / 500.0)

        # Brightness score (penalize too dark or too bright)
        mean_brightness = np.mean(gray)
        brightness_score = 1.0 - abs(mean_brightness - 127) / 127.0

        # Weighted combination
        quality = (size_score * 0.3) + (blur_score * 0.4) + (brightness_score * 0.3)
        return round(max(0.0, min(1.0, quality)), 3)

    def detect_faces_in_crop(
        self, crop: np.ndarray
    ) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
        """
        Detect faces in a pre-cropped region (e.g. from YOLO person bbox).
        Runs at full resolution for maximum sensitivity.

        Args:
            crop: BGR image crop (numpy array)

        Returns:
            Tuple of (face_locations_in_crop_coords, bgr_crop)

        Note: InsightFace expects BGR input — it does swapRB=True
        internally. Do NOT convert to RGB before calling.
        """
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            return [], crop

        # Pass BGR directly — InsightFace handles BGR→RGB internally
        app = self._get_app()
        if app is None:
            return [], crop

        faces = app.get(crop)

        face_locations = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox

            # Filter impossibly small faces
            if (y2 - y1) < 10 or (x2 - x1) < 10:
                continue

            # Convert to (top, right, bottom, left)
            face_locations.append((y1, x2, y2, x1))

        return face_locations, crop

    @staticmethod
    def estimate_head_region(
        person_bbox: dict, frame_shape: tuple
    ) -> Tuple[int, int, int, int]:
        """
        Estimate the head/upper-body region from a YOLO person bounding box.
        Returns (top, right, bottom, left) in face_recognition format.

        The head is approximately the top 35% of the person bounding box,
        centered horizontally.
        """
        x1 = person_bbox["x1"]
        y1 = person_bbox["y1"]
        x2 = person_bbox["x2"]
        y2 = person_bbox["y2"]
        h, w = frame_shape[:2]

        person_h = y2 - y1
        person_w = x2 - x1

        # Head region: top 40% of the person box, with some padding
        head_bottom = y1 + int(person_h * 0.40)
        head_top = max(0, y1 - int(person_h * 0.05))  # small pad above head
        head_left = max(0, x1 - int(person_w * 0.1))
        head_right = min(w, x2 + int(person_w * 0.1))
        head_bottom = min(h, head_bottom)

        return head_top, head_right, head_bottom, head_left

    @staticmethod
    def location_to_xyxy(location: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Convert (top, right, bottom, left) to (x1, y1, x2, y2)."""
        top, right, bottom, left = location
        return left, top, right, bottom

    @staticmethod
    def xyxy_to_location(x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int, int, int]:
        """Convert (x1, y1, x2, y2) to (top, right, bottom, left)."""
        return y1, x2, y2, x1

    @property
    def frames_processed(self) -> int:
        return self._frame_count


# ══════════════════════════════════════════════════════════════════════════════
#  Vehicle + Plate Detector — YOLO11-Large with TensorRT support
# ══════════════════════════════════════════════════════════════════════════════

# Lazy-loaded singleton for VRAM efficiency
_vehicle_plate_detector = None
_vehicle_plate_lock = None
_vehicle_plate_failed = False

# COCO vehicle class IDs (used when running COCO model as fallback)
COCO_VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def get_vehicle_plate_detector():
    """Get or create the singleton VehiclePlateDetector instance."""
    global _vehicle_plate_detector, _vehicle_plate_lock, _vehicle_plate_failed
    import threading

    if _vehicle_plate_lock is None:
        _vehicle_plate_lock = threading.Lock()

    with _vehicle_plate_lock:
        if _vehicle_plate_detector is not None:
            return _vehicle_plate_detector
        if _vehicle_plate_failed:
            return None
        try:
            _vehicle_plate_detector = VehiclePlateDetector()
            return _vehicle_plate_detector
        except Exception as e:
            logger.error(f"Failed to create VehiclePlateDetector: {e}", exc_info=True)
            _vehicle_plate_failed = True
            return None


class VehiclePlateDetector:
    """
    Unified vehicle + license plate detector using YOLO11-Large.

    Provides:
    - Single-pass detection of vehicles AND license plates at 1280px
    - TensorRT `.engine` export for RTX 5070 optimization
    - Fallback to COCO YOLOv8 for vehicle-only detection
    - GPU-optimized inference with configurable batch size

    The model outputs two class groups:
    - Vehicle classes (car, motorcycle, bus, truck, etc.)
    - License plate class (single class for plate bounding boxes)

    Hardware target: NVIDIA RTX 5070 (12GB VRAM)
    """

    def __init__(self):
        import threading

        self._vehicle_model = None        # COCO model (fallback for vehicles)
        self._plate_model = None          # Custom plate detector (YOLO11-L)
        self._unified_model = None        # Unified vehicle+plate model (future)

        self._vehicle_loaded = False
        self._plate_loaded = False
        self._vehicle_available = False
        self._plate_available = False

        self._device = 'cpu'
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()

        # Detect GPU
        try:
            import torch
            if torch.cuda.is_available():
                self._device = 'cuda'
                gpu_name = torch.cuda.get_device_name(0)
                vram_mb = torch.cuda.get_device_properties(0).total_mem // (1024 * 1024)
                logger.info(
                    f"VehiclePlateDetector: GPU={gpu_name}, "
                    f"VRAM={vram_mb}MB, device=cuda"
                )
            else:
                logger.info("VehiclePlateDetector: No CUDA GPU, using CPU")
        except ImportError:
            logger.info("VehiclePlateDetector: PyTorch not installed, using CPU")

    def load_vehicle_model(self, model_path: str = "yolov8x.pt") -> bool:
        """
        Load the COCO vehicle detection model.

        Uses YOLOv8x (largest COCO variant) for robust vehicle detection.
        With 12GB VRAM on the RTX 5070, this runs comfortably alongside
        the plate detector.

        Args:
            model_path: Path to YOLO weights (.pt or .engine)

        Returns:
            True if model loaded successfully
        """
        with self._load_lock:
            if self._vehicle_loaded:
                return self._vehicle_available
            try:
                from ultralytics import YOLO
                import os

                # Check for TensorRT engine first
                engine_path = model_path.replace('.pt', '.engine')
                if os.path.exists(engine_path):
                    logger.info(f"Loading TensorRT vehicle model: {engine_path}")
                    self._vehicle_model = YOLO(engine_path)
                else:
                    logger.info(f"Loading vehicle detection model: {model_path}")
                    self._vehicle_model = YOLO(model_path)

                self._vehicle_available = True
                logger.info(
                    f"✅ Vehicle detection model loaded on {self._device}"
                )
            except Exception as e:
                self._vehicle_available = False
                logger.error(f"Failed to load vehicle model: {e}")
            finally:
                self._vehicle_loaded = True
            return self._vehicle_available

    def load_plate_model(self, model_path: str = None) -> bool:
        """
        Load the YOLO11-Large license plate detection model.

        This is the primary plate localizer — replaces the old lightweight
        models with a deep-parameter network for robust plate detection
        under CCTV conditions (motion blur, steep angles, environmental noise).

        Args:
            model_path: Path to YOLO11-L plate weights (.pt or .engine)

        Returns:
            True if model loaded successfully
        """
        with self._load_lock:
            if self._plate_loaded:
                return self._plate_available
            try:
                from ultralytics import YOLO
                import os

                if model_path is None:
                    model_path = settings.ANPR_MODEL_PATH

                if not os.path.exists(model_path):
                    logger.warning(f"Plate model not found: {model_path}")
                    self._plate_available = False
                    self._plate_loaded = True
                    return False

                # Check for TensorRT engine first
                engine_path = model_path.replace('.pt', '.engine')
                if os.path.exists(engine_path):
                    logger.info(f"Loading TensorRT plate model: {engine_path}")
                    self._plate_model = YOLO(engine_path)
                else:
                    logger.info(f"Loading YOLO11-L plate model: {model_path}")
                    self._plate_model = YOLO(model_path)

                model_names = getattr(self._plate_model, 'names', {})
                num_classes = len(model_names) if model_names else 0
                self._plate_available = True
                logger.info(
                    f"✅ Plate detection model loaded: "
                    f"{num_classes} classes = {list(model_names.values())} "
                    f"on {self._device}"
                )
            except Exception as e:
                self._plate_available = False
                logger.error(f"Failed to load plate model: {e}")
            finally:
                self._plate_loaded = True
            return self._plate_available

    def detect_vehicles(
        self,
        frame: np.ndarray,
        conf: float = 0.25,
        imgsz: int = 1280,
    ) -> List[dict]:
        """
        Detect vehicles in a frame using the COCO model.

        Returns list of dicts with keys:
        - box: (x1, y1, x2, y2) pixel coordinates
        - type: vehicle class name (car, motorcycle, bus, truck)
        - confidence: detection confidence score
        - class_id: COCO class ID

        Args:
            frame: BGR image from OpenCV
            conf: Minimum detection confidence
            imgsz: Inference image size (higher = better for distant vehicles)
        """
        if not self.load_vehicle_model():
            return []

        try:
            results = self._vehicle_model(
                frame,
                conf=conf,
                classes=list(COCO_VEHICLE_CLASSES.keys()),
                imgsz=imgsz,
                verbose=False,
                device=self._device,
            )

            vehicles = []
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id not in COCO_VEHICLE_CLASSES:
                        continue
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = [int(c) for c in box.xyxy[0].tolist()]
                    vehicles.append({
                        'box': (x1, y1, x2, y2),
                        'type': COCO_VEHICLE_CLASSES[cls_id],
                        'confidence': confidence,
                        'class_id': cls_id,
                    })
            return vehicles

        except Exception as e:
            logger.error(f"Vehicle detection failed: {e}")
            return []

    def detect_plates(
        self,
        frame: np.ndarray,
        conf: float = 0.20,
        imgsz: int = 1280,
    ) -> List[dict]:
        """
        Detect license plates in a frame using the YOLO11-L plate model.

        Runs on the full frame at high resolution for maximum recall.
        Returns plate bounding boxes with cropped plate images ready for OCR.

        Args:
            frame: BGR image from OpenCV
            conf: Minimum detection confidence
            imgsz: Inference image size

        Returns:
            List of dicts with keys: box, confidence, crop, class_id
        """
        if not self.load_plate_model():
            return []

        try:
            results = self._plate_model(
                frame,
                conf=conf,
                imgsz=imgsz,
                verbose=False,
                device=self._device,
            )

            plates = []
            fh, fw = frame.shape[:2]
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    px1, py1, px2, py2 = [int(c) for c in box.xyxy[0].tolist()]
                    pconf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    pw, ph = px2 - px1, py2 - py1

                    # Skip impossibly small plates
                    if pw < 15 or ph < 8:
                        continue

                    # Clamp to frame bounds
                    px1, py1 = max(0, px1), max(0, py1)
                    px2, py2 = min(fw, px2), min(fh, py2)

                    plate_crop = frame[py1:py2, px1:px2]
                    if plate_crop.size == 0:
                        continue

                    plates.append({
                        'box': (px1, py1, px2, py2),
                        'confidence': pconf,
                        'crop': plate_crop,
                        'class_id': cls_id,
                    })
            return plates

        except Exception as e:
            logger.error(f"Plate detection failed: {e}")
            return []

    def detect_plates_in_crop(
        self,
        vehicle_crop: np.ndarray,
        conf: float = 0.25,
    ) -> List[dict]:
        """
        Detect plates within a vehicle crop (fallback Strategy B).

        Runs the plate detector on a cropped vehicle region for cases
        where the full-frame detection missed a plate.

        Args:
            vehicle_crop: Cropped vehicle image (BGR)
            conf: Minimum detection confidence

        Returns:
            List of plate dicts relative to the crop coordinates
        """
        if not self.load_plate_model():
            return []

        h, w = vehicle_crop.shape[:2]
        if h < 20 or w < 20:
            return []

        try:
            results = self._plate_model(
                vehicle_crop,
                conf=conf,
                verbose=False,
                device=self._device,
            )

            plates = []
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    px1, py1, px2, py2 = [int(c) for c in box.xyxy[0].tolist()]
                    pconf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    pw, ph = px2 - px1, py2 - py1

                    if pw < 15 or ph < 8:
                        continue

                    # Clamp to crop bounds
                    px1, py1 = max(0, px1), max(0, py1)
                    px2, py2 = min(w, px2), min(h, py2)

                    # Skip oversized detections (>80% of vehicle)
                    plate_area = (px2 - px1) * (py2 - py1)
                    vehicle_area = h * w
                    if vehicle_area > 0 and plate_area / vehicle_area > 0.80:
                        continue

                    plate_crop = vehicle_crop[py1:py2, px1:px2]
                    if plate_crop.size == 0:
                        continue

                    plates.append({
                        'box': (px1, py1, px2, py2),
                        'confidence': pconf,
                        'crop': plate_crop,
                        'class_id': cls_id,
                    })
            return plates

        except Exception as e:
            logger.debug(f"Plate detection in crop failed: {e}")
            return []

    def export_to_tensorrt(
        self,
        model_path: str,
        imgsz: int = 1280,
        half: bool = True,
        batch: int = 1,
    ) -> Optional[str]:
        """
        Export a YOLO .pt model to TensorRT .engine format.

        Optimized for the RTX 5070 (12GB VRAM, Ada Lovelace architecture).
        FP16 (half=True) gives ~2x speedup with minimal accuracy loss.

        Args:
            model_path: Path to the .pt weights file
            imgsz: Input image size for the engine
            half: Use FP16 precision (recommended for RTX 5070)
            batch: Batch size for inference

        Returns:
            Path to the exported .engine file, or None on failure
        """
        try:
            from ultralytics import YOLO
            import os

            if not os.path.exists(model_path):
                logger.error(f"Model not found for TensorRT export: {model_path}")
                return None

            logger.info(
                f"🚀 Starting TensorRT export: {model_path} "
                f"(imgsz={imgsz}, half={half}, batch={batch})"
            )

            model = YOLO(model_path)
            engine_path = model.export(
                format="engine",
                imgsz=imgsz,
                half=half,
                batch=batch,
                device=0,
            )

            logger.info(f"✅ TensorRT export complete: {engine_path}")
            return str(engine_path)

        except Exception as e:
            logger.error(f"TensorRT export failed: {e}", exc_info=True)
            return None

    def associate_plates_to_vehicles(
        self,
        vehicles: List[dict],
        plates: List[dict],
    ) -> List[dict]:
        """
        Associate detected plates with their parent vehicles using
        spatial proximity and containment.

        Each plate is matched to the nearest vehicle whose bounding box
        contains or is close to the plate centroid. Unmatched plates are
        returned as standalone detections.

        Args:
            vehicles: List of vehicle detection dicts
            plates: List of plate detection dicts

        Returns:
            List of association dicts with keys:
            - vehicle: vehicle dict (or None if standalone plate)
            - plate: plate dict
            - plate_crop: the plate image crop
        """
        associations = []
        matched_plate_indices = set()

        for plate_idx, plate in enumerate(plates):
            px1, py1, px2, py2 = plate['box']
            pcx = (px1 + px2) / 2
            pcy = (py1 + py2) / 2

            best_vehicle = None
            best_dist = float('inf')

            for vehicle in vehicles:
                vx1, vy1, vx2, vy2 = vehicle['box']
                # Allow a margin around the vehicle box
                margin = max(vx2 - vx1, vy2 - vy1) * 0.3
                if (vx1 - margin <= pcx <= vx2 + margin and
                        vy1 - margin <= pcy <= vy2 + margin):
                    vcx = (vx1 + vx2) / 2
                    vcy = (vy1 + vy2) / 2
                    dist = ((pcx - vcx) ** 2 + (pcy - vcy) ** 2) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_vehicle = vehicle

            associations.append({
                'vehicle': best_vehicle,
                'plate': plate,
                'plate_crop': plate['crop'],
            })
            matched_plate_indices.add(plate_idx)

        return associations

    @property
    def is_vehicle_model_available(self) -> bool:
        return self._vehicle_available

    @property
    def is_plate_model_available(self) -> bool:
        return self._plate_available

    @property
    def device(self) -> str:
        return self._device

    @property
    def stats(self) -> dict:
        return {
            "vehicle_model_available": self._vehicle_available,
            "plate_model_available": self._plate_available,
            "device": self._device,
        }
