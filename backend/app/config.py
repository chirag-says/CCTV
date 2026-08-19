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
    #
    # DISABLED BY DEFAULT. Until this is True, every face is scored at quality=0.5,
    # so _get_adaptive_threshold() always returns FACE_MATCH_BALANCED and the STRICT
    # and LENIENT tiers below are unreachable. That was previously an accidental
    # dead-code bug; it is now an explicit, reversible switch.
    #
    # Turning this on CHANGES RECOGNITION BEHAVIOUR: sharp/large faces get the
    # stricter 0.52 ceiling (fewer false accepts) and small/blurry ones get 0.72
    # (fewer false rejects). Measure FAR/FRR on a labelled set before enabling.
    # Cost when enabled: one cvtColor + Laplacian per recognised face per frame.
    ENABLE_ADAPTIVE_THRESHOLDS: bool = False
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

    # ── Identity Ledger (Phase 1) ─────────────────────────────
    # The identity layer above ByteTrack. When the tracker recreates a track,
    # the ledger decides whether it is a returning person or a new one.
    # MILESTONE 1: these only affect measurement -- nothing is assigned yet.
    IDENTITY_RETIRE_FRAMES: int = 60             # Detection-frames a vanished identity stays recoverable
    IDENTITY_LEDGER_MAX_ENTRIES: int = 50        # Hard cap on the retirement buffer (bounded memory)
    IDENTITY_MAX_RECOVERY_DISTANCE_PX: float = 400.0  # Beyond this, not a plausible predecessor

    # ── Identity Recovery (Milestone 2) ───────────────────────
    # The ledger PROPOSES a recovered identity; FaceRecognizer VERIFIES it.
    # The ledger is never the authority — it cannot assign an identity the
    # recogniser would not independently agree with.
    IDENTITY_RECOVERY_ENABLED: bool = True

    # Verification is deliberately STRICTER than ordinary recognition. A false
    # accept here restores a confident, locked identity onto the wrong person,
    # which is worse than an ordinary false accept: the identity lock then
    # suppresses re-checks for RECOGNITION_FULL_RECHECK_SEC.
    IDENTITY_VERIFY_MAX_DISTANCE: float = 0.55   # vs FACE_MATCH_BALANCED 0.62
    IDENTITY_VERIFY_MIN_MARGIN: float = 0.12     # vs FACE_MATCH_MIN_MARGIN 0.10

    # Only propose when exactly one retired identity is eligible. Milestone 1
    # measured 1 candidate for genuine recoveries and 2 for crossing people,
    # so candidate count is a usable ambiguity signal.
    IDENTITY_REQUIRE_UNAMBIGUOUS: bool = True

    # ── Unknown Registry (Phase 2) ────────────────────────────
    # Stable identities for unenrolled people. The registry is seeded from the
    # unknown_faces table so an id survives a restart and is identical across
    # cameras, instead of a fresh uuid4() per sighting session.
    UNKNOWN_REGISTRY_MAX_ENTRIES: int = 500      # LRU cap (was unbounded)
    UNKNOWN_REGISTRY_REFRESH_SEC: float = 60.0   # Reload pending unknowns every N sec
    UNKNOWN_SIGHTING_COOLDOWN_SEC: float = 30.0  # Min gap between occurrence bumps,
                                                  # so `occurrence` counts visits not frames

    # ── Unknown Re-identification (Phase 3) ───────────────────
    # An unknown identity keeps several views and matches best-of-K, mirroring
    # the enrolled gallery. With a single frozen view, a stranger first seen
    # frontally minted a SECOND id the moment they turned -- measured as 4,129
    # of 4,452 pending unknowns sitting at occurrence=1.
    UNKNOWN_MAX_VIEWS: int = 5                   # Embeddings retained per unknown
    UNKNOWN_LEARN_MIN_DISTANCE: float = 0.30     # A matched sighting becomes a new
                                                  # view only if it is at least this
                                                  # far from every view already held

    # Body-only detections (person seen, face undetectable) have no embedding,
    # so they can never be deduped, re-identified, or enrolled. Persisting them
    # produced 38% of the unknown_faces table as permanent unmergeable clutter.
    # They are still detected, tracked, and drawn -- just not given an identity
    # row. Set true to restore the previous behaviour.
    PERSIST_BODY_ONLY_UNKNOWNS: bool = False

    # ── Unknown Maturation Gate ────────────────────────────────
    # A track must remain unrecognised for this many seconds before
    # an unknown snapshot is persisted. Prevents premature unknowns
    # when recognition hasn't stabilised yet (e.g. the very first frame
    # a face appears, before hysteresis / temporal voting can act).
    # Time-based (not frame-based) so behaviour is FPS-independent.
    UNKNOWN_MATURATION_SEC: float = 0.5

    # Shadow Gallery (background validation)
    SHADOW_GALLERY_INTERVAL_MIN: int = 30    # How often to re-validate (minutes)
    SHADOW_GALLERY_ENABLED: bool = True      # Auto-start shadow gallery on boot

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

    # ── Stream Health & Auto-Recovery ─────────────────────────
    STREAM_STALE_TIMEOUT_SEC: float = 10.0       # No fresh frame for this long → STALE
    STREAM_RECONNECT_INITIAL_SEC: float = 2.0    # First reconnect wait
    STREAM_RECONNECT_MAX_SEC: float = 60.0       # Max backoff ceiling
    STREAM_RECONNECT_BACKOFF: float = 2.0        # Exponential backoff multiplier
    STREAM_FROZEN_HASH_TIMEOUT_SEC: float = 15.0 # Identical frame hash for this long → STALE
    STREAM_MAX_RECONNECT_ATTEMPTS: int = 100     # Give up after N attempts → OFFLINE

    # ── AI Identity Review (Background Learning) ─────────────
    ENABLE_IDENTITY_REVIEW: bool = False          # Master switch
    IDENTITY_REVIEW_INTERVAL_SEC: int = 300       # Run clustering every 5 min
    IDENTITY_REVIEW_MIN_CLUSTER_SIZE: int = 3     # Min images to form a cluster
    IDENTITY_REVIEW_TOP_K: int = 5                # Candidate persons per cluster
    IDENTITY_REVIEW_CLUSTER_THRESHOLD: float = 0.55  # Agglomerative clustering distance threshold
    IDENTITY_REVIEW_VLM_PROVIDER: str = "ollama"  # ollama/openai/disabled
    IDENTITY_REVIEW_VLM_MODEL: str = "gpt-4o"     # Only used when provider=openai
    IDENTITY_REVIEW_VLM_API_KEY: str = ""          # Only used when provider=openai
    IDENTITY_REVIEW_REQUIRE_MULTI_FRAME: bool = True  # Require seen across multiple frames

    # Three-Tier Candidate Pipeline
    # Safety-valve review window: candidates outside this range are auto-rejected
    # or auto-queued. Change these without rewriting logic.
    IDENTITY_REVIEW_REVIEW_MIN: float = 0.40     # Below this → reject immediately
    IDENTITY_REVIEW_REVIEW_MAX: float = 0.98     # Above this → auto-queue without VLM
    # Tier thresholds (within the review window):
    IDENTITY_REVIEW_TIER_HIGH: float = 0.85      # >= this → skip VLM, create candidate directly
    IDENTITY_REVIEW_TIER_LOW: float = 0.40       # >= this → send to VLM for verification
    #                                              < TIER_LOW → reject (must equal REVIEW_MIN)
    # VLM budget & fairness
    IDENTITY_REVIEW_VLM_MAX_PER_CYCLE: int = 30  # Max VLM evaluations per worker cycle
    IDENTITY_REVIEW_MAX_CLUSTERS_PER_PERSON: int = 3  # Per-person fairness cap per cycle

    # ── Ollama Local VLM ──────────────────────────────────────
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5vl:7b"
    OLLAMA_TIMEOUT_SEC: float = 120.0             # VLM inference timeout (seconds)

    # ── Gallery Optimizer ─────────────────────────────────────
    GALLERY_OPT_MIN_FACE_PX: int = 60             # Skip crops smaller than this
    GALLERY_OPT_MIN_BLUR_SCORE: float = 40.0      # Skip blurry crops
    GALLERY_OPT_MIN_DET_SCORE: float = 0.45       # Skip low-confidence detections
    GALLERY_OPT_OUTLIER_FACTOR: float = 1.5       # Cluster outlier threshold (× median distance)
    GALLERY_OPT_VLM_CACHE_TTL_HOURS: int = 168    # Cache VLM results for 7 days

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
