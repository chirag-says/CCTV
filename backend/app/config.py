"""
Application configuration using Pydantic Settings.
Loads from environment variables / .env file.
"""

from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    # ── PostgreSQL Database ───────────────────────────────────
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "sentinelai"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "postgres"
    DB_ECHO: bool = False  # Set True to log all SQL queries

    # ── JWT Auth ──────────────────────────────────────────────
    JWT_SECRET_KEY: str = ""  # MUST be set via .env — no insecure default
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # ── Vision Pipeline ───────────────────────────────────────
    # Face Detection (SCRFD via insightface)
    INSIGHTFACE_MODEL_NAME: str = "buffalo_l"  # buffalo_l has SCRFD_10G + ArcFace R50
    INSIGHTFACE_DET_SIZE: int = 1024  # Detection input size (1024x1024, was 640)
    # 1024 gives ~2.5x better resolution for small faces on 4MP frames.
    # On a 2560x1440 source, 1024 letterbox means 0.71x downscale
    # (vs 0.44x at 640), so a 60px face arrives at SCRFD as 43px not 26px.
    INSIGHTFACE_DET_THRESH: float = 0.5  # Face detection confidence threshold

    # Face Recognition (ArcFace via insightface)
    FACE_MATCH_TOLERANCE: float = 0.75  # Legacy fallback cosine distance threshold
    FACE_EMBEDDING_DIM: int = 512  # ArcFace embedding dimension

    # Adaptive thresholds (initial defaults — calibrate per deployment via FAR/FRR benchmarking)
    FACE_MATCH_STRICT: float = 0.52     # High-quality face threshold (quality >= 0.8)
    FACE_MATCH_BALANCED: float = 0.62   # Medium-quality threshold (quality >= 0.5)
    FACE_MATCH_LENIENT: float = 0.72    # Low-quality threshold (quality < 0.5)
    FACE_MATCH_MIN_MARGIN: float = 0.10 # Min gap between best and second-best person match
    FACE_MATCH_TOP_K: int = 5           # Number of candidate persons to retrieve

    # Face Quality Gate (minimum thresholds for enrollment)
    FACE_QUALITY_MIN_BLUR: float = 80.0       # Minimum Laplacian variance (higher = sharper)
    FACE_QUALITY_MIN_BRIGHTNESS: float = 40.0 # Minimum mean pixel brightness
    FACE_QUALITY_MAX_BRIGHTNESS: float = 220.0  # Maximum mean pixel brightness
    FACE_QUALITY_MIN_SIZE: int = 80            # Minimum face width/height in pixels
    FACE_QUALITY_MIN_DET_SCORE: float = 0.55   # Minimum SCRFD detection confidence
    FACE_QUALITY_MAX_YAW: float = 45.0         # Maximum head yaw (degrees)
    FACE_QUALITY_MAX_PITCH: float = 35.0       # Maximum head pitch (degrees)

    # FP16 Inference (enable only after benchmarking accuracy)
    ENABLE_FP16: bool = False

    # Search Backend: "numpy" (default) or "faiss" (future)
    SEARCH_BACKEND: str = "numpy"

    # Embedding Suggestion System
    # "Learn coverage, not accumulate images" — maintain a small, diverse,
    # high-quality representation that maximizes recognition accuracy.
    MAX_ENCODINGS_PER_PERSON: int = 15         # Hard cap on encodings per person
    SUGGESTION_MIN_CONFIDENCE: float = 0.85    # Only suggest if match confidence >= this
    SUGGESTION_DUPLICATE_THRESHOLD: float = 0.15  # Skip if cosine distance < this from any existing
    SUGGESTION_COOLDOWN_SEC: int = 60          # Max 1 suggestion per person per N seconds
    SUGGESTION_QUALITY_MARGIN: float = 0.10    # Quality improvement needed to suggest "better quality"
    SUGGESTION_SHARPNESS_MARGIN: float = 1.3   # Blur score multiplier to qualify as "sharper"

    # Pose Buckets (yaw boundaries in degrees)
    # Frontal: -15° to +15°  |  Left: 15° to 45°  |  Extreme Left: 45°+
    # Right: -45° to -15°    |  Extreme Right: < -45°
    POSE_BUCKET_FRONTAL: float = 15.0    # |yaw| <= this → "frontal"
    POSE_BUCKET_MODERATE: float = 45.0   # this > |yaw| > FRONTAL → "left"/"right"
    # |yaw| > MODERATE → "extreme_left"/"extreme_right"

    # ByteTrack Multi-Object Tracking
    BYTETRACK_TRACK_THRESH: float = 0.5  # High-confidence detection threshold
    BYTETRACK_TRACK_BUFFER: int = 90  # Frames to keep lost tracks alive (~3s at 30fps, prevents track fragmentation)
    BYTETRACK_MATCH_THRESH: float = 0.8  # IoU threshold for matching
    BYTETRACK_TRACKED_GRACE: int = 2  # Detection-frames before tracked→lost (increase to reduce fragmentation)

    # Recognition Cache (time-based, FPS-independent)
    # Once a track is confidently recognized, skip ArcFace until re-check
    RECOGNITION_LOCK_CONFIDENCE: float = 0.85   # Min confidence to lock a track identity
    RECOGNITION_RECHECK_SEC: float = 1.0         # Re-check unlocked tracks every N seconds
    RECOGNITION_FULL_RECHECK_SEC: float = 3.0    # Re-check locked tracks every N seconds

    # Identity Hysteresis (anti-flicker for borderline embeddings)
    # When a known person's distance oscillates around the threshold,
    # hysteresis prevents Known→Unknown→Known flicker by requiring
    # N consecutive failures before dropping identity.
    IDENTITY_GRACE_FRAMES: int = 5               # Consecutive failures before dropping identity
    IDENTITY_GRAY_ZONE_UPPER: float = 0.08       # Distance above threshold that's still "gray zone"
                                                  # e.g., balanced=0.62 → gray zone extends to 0.70
                                                  # Distances above gray ceiling = immediate unknown

    # Async Event Queue
    EVENT_QUEUE_SIZE: int = 500                  # Max queued events (critical events bypass this limit)

    # Pipeline Benchmarks
    ENABLE_PIPELINE_BENCHMARKS: bool = True       # Log per-stage timing every 100 frames

    # Performance Profiler (structured aggregate reports)
    ENABLE_PROFILING: bool = True                 # Enable performance profiler (disable in production)
    DEBUG_LOGS: bool = False                      # Enable DEBUG-level app logs (independent of uvicorn reload)

    FRAME_SKIP: int = 3
    DETECTION_SCALE: float = 0.5
    EXIT_THRESHOLD_SECONDS: int = 30  # 30 seconds — avoids premature exits from brief occlusions
    ENTRY_THRESHOLD_SECONDS: int = 3
    UNKNOWN_SIMILARITY_THRESHOLD: float = 0.70  # Cosine distance for unknown dedup (lower = merges more aggressively, reduces duplicates)

    # Face size thresholds — split detection gate from recognition gate
    # Detection gate: small faces are useful for tracking, counting, entry/exit
    # Recognition gate: only compute embeddings on faces large enough for reliable matching
    MIN_FACE_DETECT_PX: int = 32     # Minimum face size to detect and track
    MIN_FACE_RECOGNIZE_PX: int = 80  # Minimum face size to extract embeddings (recognition)

    # ── Camera ────────────────────────────────────────────────
    DEFAULT_CAMERA_SOURCE: str = "0"
    MAX_CAMERAS: int = 10

    # ── Server ────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:3001"

    # ── Storage ───────────────────────────────────────────────
    SNAPSHOT_DIR: str = "./snapshots"
    MAX_SNAPSHOT_SIZE_KB: int = 200

    # ── Safety & Security Modules ─────────────────────────────
    # Attribute Recognition
    ENABLE_ATTRIBUTE_RECOGNITION: bool = True

    # Crowd / Gathering Detection
    ENABLE_CROWD_DETECTION: bool = True
    CROWD_PROXIMITY_PX: int = 150       # ~1.5m at typical camera distance
    CROWD_MIN_PERSONS: int = 3
    CROWD_SUSTAIN_SECONDS: float = 5.0

    # Loitering / Idle Detection
    ENABLE_LOITERING_DETECTION: bool = True
    LOITER_MOVEMENT_THRESHOLD_PX: int = 50
    LOITER_TIME_WINDOW_SEC: float = 300.0    # 5 minutes
    LOITER_ALERT_COOLDOWN_SEC: float = 600.0 # 10 min re-alert cooldown

    # Hazard Detection (YOLOv8)
    ENABLE_HAZARD_DETECTION: bool = True
    HAZARD_MODEL_PATH: str = "yolov8n.pt"
    HAZARD_FRAME_INTERVAL: int = 30      # Process every 30th frame
    HAZARD_ALERT_COOLDOWN_SEC: float = 30.0

    # ── Pipeline Mode ─────────────────────────────────────────
    # "face" = Face Recognition mode (default, existing behaviour)
    # "traffic" = Traffic mode (ANPR + vehicle-person safety)
    PIPELINE_MODE: str = "hybrid"

    # ── RF-DETR (Real-Time Detection Transformer) ─────────────
    RFDETR_MODEL_SIZE: str = "large"          # "nano", "base", or "large"
    RFDETR_CONFIDENCE_THRESHOLD: float = 0.40
    RFDETR_DEVICE: str = "auto"               # "auto", "cuda", or "cpu"

    # ── Traffic Mode: ANPR (License Plate Recognition) ────────
    ENABLE_ANPR: bool = True
    ANPR_MODEL_PATH: str = "./models/license-plate-finetune-v1l.pt"
    ANPR_FRAME_INTERVAL: int = 5         # Process every 5th frame
    ANPR_PLATE_COOLDOWN_SEC: float = 60.0
    ANPR_OCR_LANGUAGES: str = "en"       # Comma-separated EasyOCR languages
    ANPR_VEHICLE_MODEL: str = "yolov8x.pt"  # COCO model for vehicle detection (yolov8x for RTX 5070)
    ANPR_VEHICLE_IMGSZ: int = 1280       # Inference size for vehicle/plate detection
    ANPR_KEYFRAME_MIN_FRAMES: int = 3    # Min frames before OCR trigger per vehicle
    ANPR_KEYFRAME_MAX_FRAMES: int = 15   # Max frames for OCR accumulation

    # ── Traffic Mode: Vehicle-Person Safety Monitor ───────────
    ENABLE_TRAFFIC_MONITOR: bool = True
    TRAFFIC_MODEL_PATH: str = "./models/gate_vehicles_v2.pt"
    TRAFFIC_FRAME_INTERVAL: int = 5      # Process every 5th frame
    TRAFFIC_PROXIMITY_PX: int = 50       # Pixel proximity for danger alert
    TRAFFIC_ALERT_COOLDOWN_SEC: float = 15.0

    # ── Roboflow Indian Vehicles Detection API ────────────────
    ROBOFLOW_API_KEY: str = ""  # MUST be set via .env
    ROBOFLOW_MODEL_ID: str = "indian-vehicles-dataset/2"
    ROBOFLOW_API_URL: str = "https://serverless.roboflow.com"
    ROBOFLOW_CONFIDENCE_THRESHOLD: float = 0.50
    ENABLE_ROBOFLOW_DETECTION: bool = True  # Use Roboflow as primary vehicle detector

    # ── Roboflow Person Detection API ─────────────────────────
    ROBOFLOW_PERSON_MODEL_ID: str = "people-detection-o4rdr/10"
    ROBOFLOW_PERSON_CONFIDENCE: float = 0.50
    ENABLE_ROBOFLOW_PERSON: bool = True  # Use Roboflow person model for person detection

    # ── Video Analysis ────────────────────────────────────────
    VIDEO_UPLOAD_DIR: str = "./uploads/videos"
    MAX_VIDEO_SIZE_MB: int = 2000
    ALLOWED_VIDEO_EXTENSIONS: str = "mp4,avi,mkv,mov,wmv,flv,webm"
    VIDEO_ANALYSIS_FRAME_SKIP: int = 5   # Process every 5th frame for speed
    VIDEO_ANALYSIS_MAX_CONCURRENT: int = 2

    # ── Person Re-Identification (ReID) ───────────────────────
    ENABLE_REID: bool = True
    REID_MATCH_THRESHOLD: float = 0.65    # Cosine similarity (0-1, higher = stricter)
    REID_GALLERY_MAX_SIZE: int = 500      # Max tracked persons before LRU eviction
    REID_SIGHTING_COOLDOWN: float = 5.0   # Min seconds between sightings per person/camera
    REID_FRAME_INTERVAL: int = 10         # Extract ReID features every N frames (performance)

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()

# Ensure snapshot directory exists
os.makedirs(settings.SNAPSHOT_DIR, exist_ok=True)
# Ensure video upload directory exists
os.makedirs(settings.VIDEO_UPLOAD_DIR, exist_ok=True)
