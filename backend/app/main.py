"""
AI-Powered CCTV Surveillance System — FastAPI Application Entry Point.

Features:
- Face detection & recognition from live video streams
- Entry/Exit tracking with time-based logic
- Unknown face detection & admin enrollment queue
- Real-time WebSocket events
- Analytics dashboard API
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os

from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.services.person_service import PersonService
from app.vision.camera_worker import camera_manager
from app.core.websocket import ws_manager
from app.db.session import SessionLocal, init_db, create_default_admin
from app.db.models import DetectionEvent, TrackingSession, UnknownFace
from app.utils.image_utils import frame_to_jpeg_bytes, resize_frame

# ── Logging Setup ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG_LOGS else logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)-25s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cctv")

# Suppress noisy third-party loggers (binary WebSocket dumps, per-request logs)
# These MUST be WARNING regardless of DEBUG_LOGS to prevent log flooding.
for _noisy in (
    "websockets", "websockets.protocol", "websockets.server",
    "uvicorn.access", "uvicorn.error",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# Module-level reference to the main asyncio event loop (set during startup)
_main_loop: asyncio.AbstractEventLoop = None


# ── Event Handlers (Pipeline → EventQueue → DB/WS) ───────────
#
# The vision pipeline calls handle_pipeline_event() and handle_unknown_face()
# from the camera worker thread. These functions MUST be non-blocking.
# They enqueue events into the EventQueue which processes them in a
# dedicated background consumer thread.

from app.core.event_queue import get_event_queue, EventPriority
from app.core.profiler import profiler


async def _broadcast_event(event: dict):
    """Forward pipeline event to WebSocket clients."""
    await ws_manager.broadcast_event(event)


def handle_frame(camera_id: str, frame):
    """
    Legacy callback — JPEG encoding is now done in the camera worker thread
    and cached as pipeline._last_jpeg. The WebSocket stream endpoint reads
    those pre-encoded bytes directly. This callback is no longer needed.
    """
    pass


def handle_pipeline_event(event: dict):
    """
    Called by vision pipeline on entry/exit/detection events.
    NON-BLOCKING: enqueues event for async processing.
    """
    event_type = event.get('event_type', 'unknown')

    # vehicle_count events are high-frequency — broadcast only, don't persist
    if event_type == "vehicle_count":
        if _main_loop:
            try:
                asyncio.run_coroutine_threadsafe(_broadcast_event(event), _main_loop)
            except Exception:
                pass
        return

    # Critical events (entry, exit, security alerts) must never be dropped
    is_critical = event_type in ("entry", "exit", "security_alert")
    priority = EventPriority.CRITICAL if is_critical else EventPriority.NORMAL

    eq = get_event_queue()
    with profiler.measure("queue_put"):
        eq.put("pipeline_event", event, priority=priority)


def handle_unknown_face(data: dict):
    """
    Called when an unknown face is detected.
    NON-BLOCKING: enqueues for async processing (snapshot saving + DB + WS).
    """
    eq = get_event_queue()
    with profiler.measure("queue_put"):
        eq.put("unknown_face", data, priority=EventPriority.NORMAL)


# ── Actual Event Processing (runs in EventQueue consumer thread) ──────────

def _process_pipeline_event(event: dict):
    """
    Actual handler for pipeline events. Runs in the EventQueue consumer thread.
    Persists to PostgreSQL and broadcasts via WebSocket.
    """
    event_type = event.get('event_type', 'unknown')
    logger.info(f"📡 Event: {event_type} — {event.get('person_name', 'N/A')}")

    eq = get_event_queue()

    # Persist to PostgreSQL
    import time as _time
    _db_start = _time.perf_counter()
    _db_ok = False
    try:
        from uuid import uuid4
        from datetime import datetime, timezone

        db = SessionLocal()
        try:
            camera_id = event.get("camera_id", "")

            # Browser cameras don't have a DB record — create one on-the-fly
            if camera_id.startswith("bc_"):
                from app.db.models import Camera
                existing = db.query(Camera).filter(Camera.id == camera_id).first()
                if not existing:
                    cam = Camera(
                        id=camera_id,
                        name="Remote Camera",
                        location="Browser",
                        stream_url=f"browser:{camera_id}",
                        camera_type="browser",
                        is_active=True,
                    )
                    db.add(cam)
                    db.flush()

            # ── FK Guard: use in-memory person ID cache instead of DB query ──
            person_id = event.get("person_id")
            event_metadata = {
                k: v for k, v in event.items()
                if k not in ("person_id", "camera_id", "event_type", "confidence", "snapshot_url", "subtype")
            }
            if person_id and not eq.is_known_person(person_id):
                if str(person_id).startswith("anon:"):
                    # A track handle, not an identity: a face too small to
                    # embed. It counts toward presence but has no identity, so
                    # recording the ByteTrack id would put an ephemeral internal
                    # handle into the permanent event log.
                    event_metadata["anonymous"] = True
                else:
                    # Stable unknown identity — no FK (no persons row), so it
                    # lives in metadata. This id is now consistent across
                    # restarts and cameras, so the same stranger is traceable.
                    event_metadata["unknown_person_id"] = person_id
                person_id = None

            # Create detection event
            db_event = DetectionEvent(
                id=str(uuid4()),
                person_id=person_id,
                camera_id=camera_id,
                event_type=event.get("event_type", "detection"),
                subtype=event.get("subtype"),
                confidence=event.get("confidence", 0.0),
                snapshot_url=event.get("snapshot_url"),
                metadata_json=event_metadata,
                created_at=datetime.fromisoformat(event["timestamp"]) if event.get("timestamp") else datetime.now(timezone.utc),
            )
            db.add(db_event)

            # Create/update tracking session for entry/exit (only for known persons)
            if event["event_type"] == "entry" and person_id:
                session = TrackingSession(
                    id=str(uuid4()),
                    person_id=person_id,
                    camera_id=event["camera_id"],
                    entry_time=datetime.fromisoformat(event["timestamp"]) if event.get("timestamp") else datetime.now(timezone.utc),
                    status="active",
                    created_at=datetime.now(timezone.utc),
                )
                db.add(session)
                db.commit()

                # Set session ID on tracker
                for cid in camera_manager.get_active_camera_ids():
                    worker = camera_manager._workers.get(cid)
                    if worker:
                        worker.pipeline.tracker.set_session_id(
                            event.get("person_id", person_id), session.id
                        )
            elif event["event_type"] == "exit" and event.get("session_id"):
                session = db.query(TrackingSession).filter(
                    TrackingSession.id == event["session_id"]
                ).first()
                if session:
                    session.exit_time = datetime.fromisoformat(event["timestamp"]) if event.get("timestamp") else datetime.now(timezone.utc)
                    session.duration_sec = event.get("duration_sec", 0)
                    session.status = "completed"

            db.commit()
            _db_ok = True
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to persist event: {e}")
        finally:
            db.close()

    except Exception as e:
        logger.error(f"Failed to persist event: {e}")

    # Record DB timing in profiler
    _db_elapsed = (_time.perf_counter() - _db_start) * 1000
    if _db_ok:
        profiler.record_db_insert(_db_elapsed)
    else:
        profiler.record_db_failure()

    # Broadcast via WebSocket (fire-and-forget async)
    if _main_loop:
        try:
            asyncio.run_coroutine_threadsafe(_broadcast_event(event), _main_loop)
        except Exception:
            pass


def _process_unknown_face(data: dict):
    """
    Actual handler for unknown face events. Runs in EventQueue consumer thread.

    Handles:
    1. Saving snapshots to disk (if frame data is provided)
    2. DB-level dedup via encoding comparison
    3. Persisting to unknown_faces table
    4. Broadcasting via WebSocket
    """
    detection_type = data.get("detection_type", "face")

    # ── Repeat sighting of an already-persisted identity ──
    # Carries no crops: the row already has snapshots. Only the visit counters
    # move. Previously every repeat wrote three JPEGs that were then orphaned
    # when the dedup merged the sighting into the existing row.
    if data.get("sighting_only"):
        from datetime import datetime as _dt, timezone as _tz
        from app.services.unknown_service import UnknownService
        ts = (
            _dt.fromisoformat(data["timestamp"]) if data.get("timestamp")
            else _dt.now(_tz.utc)
        )
        if UnknownService.record_sighting(data["unknown_id"], data.get("camera_id", ""), ts):
            logger.debug(
                f"👤 Unknown {data['unknown_id'][:8]}… seen again on "
                f"camera {data.get('camera_id')}"
            )
        return

    logger.info(
        f"👤 Unknown {detection_type} detected on camera {data.get('camera_id')}"
    )

    # ── Snapshots: DEFERRED until we know a row will be created ──
    # These used to be written eagerly, before the dedup query below. When the
    # dedup then merged the sighting into an existing row, all three JPEGs were
    # orphaned on disk with nothing referencing them. Saving is now a closure
    # invoked only on the create path.
    import time as _time
    _paths = {
        "snapshot": data.get("snapshot_path", ""),
        "context": data.get("context_path", ""),
        "full_frame": data.get("full_frame_path", ""),
    }

    def _save_snapshots():
        """Write the three crops to disk. Only called when inserting a row."""
        if data.get("face_crop") is None:
            return
        _snap_start = _time.perf_counter()
        try:
            from app.utils.image_utils import save_snapshot
            unknown_id = data.get("unknown_id", "")

            _, _paths["snapshot"] = save_snapshot(
                data["face_crop"], f"unknown_{unknown_id}",
                subdirectory="unknowns", quality=95,
            )
            if data.get("context_crop") is not None:
                _, _paths["context"] = save_snapshot(
                    data["context_crop"], f"context_{unknown_id}",
                    subdirectory="unknowns", quality=95,
                )
            if data.get("full_frame") is not None:
                _, _paths["full_frame"] = save_snapshot(
                    data["full_frame"], f"fullframe_{unknown_id}",
                    subdirectory="frames", quality=90,
                )
            profiler.record_snapshot_save((_time.perf_counter() - _snap_start) * 1000)
        except Exception as e:
            logger.error(f"Failed to save unknown snapshots: {e}")

    try:
        import numpy as np
        from uuid import uuid4
        from datetime import datetime, timezone
        from app.vision.recognizer import serialize_encoding, deserialize_encoding

        db = SessionLocal()
        try:
            # Handle encoding: may be None for body-only detections
            encoding_bytes = None
            encoding_array = None
            if data.get("encoding") is not None:
                encoding_array = data["encoding"]
                encoding_bytes = serialize_encoding(encoding_array)  # raw float32 bytes

            ts = datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now(timezone.utc)
            camera_id = data.get("camera_id", "")

            # Browser cameras don't have a DB record — create one on-the-fly
            if camera_id.startswith("bc_"):
                from app.db.models import Camera
                existing = db.query(Camera).filter(Camera.id == camera_id).first()
                if not existing:
                    cam = Camera(
                        id=camera_id,
                        name="Remote Camera",
                        location="Browser",
                        stream_url=f"browser:{camera_id}",
                        camera_type="browser",
                        is_active=True,
                    )
                    db.add(cam)
                    db.flush()

            # ── DB-level dedup: check existing pending unknowns ──
            matched_existing = None
            if encoding_array is not None:
                # Normalize the incoming encoding
                norm_q = np.linalg.norm(encoding_array)
                if norm_q > 0:
                    enc_normalized = encoding_array / norm_q

                    # Query all pending unknown faces that have encodings
                    pending_faces = (
                        db.query(UnknownFace)
                        .filter(
                            UnknownFace.status == "pending",
                            UnknownFace.face_encoding.isnot(None),
                        )
                        .all()
                    )

                    best_match = None
                    best_distance = float("inf")
                    dedup_threshold = settings.UNKNOWN_SIMILARITY_THRESHOLD

                    for face in pending_faces:
                        try:
                            stored_enc = deserialize_encoding(face.face_encoding)
                            if stored_enc is None:
                                continue
                            norm_s = np.linalg.norm(stored_enc)
                            if norm_s > 0:
                                stored_normalized = stored_enc / norm_s
                                similarity = float(np.dot(enc_normalized, stored_normalized))
                                distance = 1.0 - similarity
                                if distance < best_distance:
                                    best_distance = distance
                                    best_match = face
                        except Exception:
                            continue

                    if best_match is not None and best_distance < dedup_threshold:
                        matched_existing = best_match
                        logger.info(
                            f"🔄 Unknown face matched existing DB entry "
                            f"{best_match.id[:12]}... (distance={best_distance:.4f}, "
                            f"occ={best_match.occurrence} → {best_match.occurrence + 1})"
                        )

            if matched_existing:
                # Another camera (or an earlier run) already has this person.
                # Bump the visit counters and reuse their snapshots — no new
                # files are written.
                matched_existing.occurrence += 1
                matched_existing.last_seen = ts
                db.commit()
                _paths["snapshot"] = matched_existing.snapshot_url or ""
                _paths["full_frame"] = matched_existing.full_frame or ""
                logger.info(
                    f"✅ Updated unknown face {matched_existing.id[:12]}... "
                    f"occurrence={matched_existing.occurrence} "
                    f"(snapshots reused, none written)"
                )
            else:
                # Genuinely new person — now it is worth the disk write.
                _save_snapshots()
                unknown = UnknownFace(
                    id=data.get("unknown_id", str(uuid4())),
                    camera_id=camera_id,
                    snapshot_url=_paths["snapshot"],
                    full_frame=_paths["full_frame"],
                    face_encoding=encoding_bytes,
                    occurrence=1,
                    first_seen=ts,
                    last_seen=ts,
                    status="pending",
                    created_at=datetime.now(timezone.utc),
                )
                db.add(unknown)
                db.commit()
                logger.info(f"✅ New unknown face persisted: {unknown.id[:12]}...")

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to persist unknown face: {e}", exc_info=True)
        finally:
            db.close()

    except Exception as e:
        logger.error(f"Failed to persist unknown face: {e}")

    # Broadcast unknown event via WebSocket
    _process_pipeline_event({
        "event_type": "unknown",
        "camera_id": data.get("camera_id", ""),
        "timestamp": data.get("timestamp", ""),
        "snapshot_url": _paths["snapshot"],
        "detection_type": detection_type,
    })


# ── Application Lifespan ─────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    global _main_loop
    _main_loop = asyncio.get_event_loop()

    logger.info("=" * 60)
    logger.info("🚀 AI CCTV Surveillance System Starting...")
    logger.info("=" * 60)

    # Initialize PostgreSQL database (create tables if needed)
    try:
        init_db()
        create_default_admin()
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        logger.error("Make sure PostgreSQL is running and accessible")
        raise

    # ── Initialize Async Event Queue ──────────────────────────
    eq = get_event_queue()
    eq.register_handler("pipeline_event", _process_pipeline_event)
    eq.register_handler("unknown_face", _process_unknown_face)
    eq.refresh_person_ids()  # Load known person IDs into memory
    eq.start()
    logger.info(f"✅ Event Queue started (size={settings.EVENT_QUEUE_SIZE})")

    # Set up pipeline callbacks (event + unknown + frame streaming)
    camera_manager.set_callbacks(
        on_event=handle_pipeline_event,
        on_unknown=handle_unknown_face,
        on_frame=handle_frame,
    )

    # Load face encodings into memory
    try:
        encodings = PersonService.get_all_encodings()
        if encodings:
            camera_manager.load_encodings_all(encodings)
            logger.info(f"✅ Loaded {len(encodings)} face encodings")
        else:
            logger.info("ℹ️  No face encodings found in database")
    except Exception as e:
        logger.warning(f"⚠️  Could not load encodings: {e}")

    logger.info(f"🌐 API Server: http://{settings.HOST}:{settings.PORT}")
    logger.info(f"📖 Docs: http://{settings.HOST}:{settings.PORT}/docs")
    logger.info("📷 Cameras will start when triggered from the dashboard")

    # ── Print Startup Summary ─────────────────────────────────
    from app.core.profiler import PerformanceProfiler
    PerformanceProfiler.print_startup_summary()

    yield  # Application is running

    # Shutdown
    logger.info("🛑 Shutting down...")
    camera_manager.stop_all()
    eq.stop()  # Drain remaining events before exit
    logger.info("✅ All cameras stopped. Event queue drained. Goodbye!")


# ── Create Application ────────────────────────────────────────

app = FastAPI(
    title="AI CCTV Surveillance System",
    description=(
        "Production-grade AI-powered video surveillance with face detection, "
        "recognition, entry/exit tracking, unknown face management, and analytics."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Request Logging Middleware ─────────────────────────────────
# Pure ASGI middleware — does NOT use BaseHTTPMiddleware to avoid
# the response-header-stripping bug that breaks CORSMiddleware.
from app.core.middleware import RequestLoggingMiddleware
app.add_middleware(RequestLoggingMiddleware)

# ── CORS ──────────────────────────────────────────────────────
# Added last so it runs FIRST (Starlette LIFO order).
# Must intercept OPTIONS preflight before any other middleware.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static Files (Snapshots) ─────────────────────────────────
os.makedirs(settings.SNAPSHOT_DIR, exist_ok=True)
app.mount("/snapshots", StaticFiles(directory=settings.SNAPSHOT_DIR), name="snapshots")

# ── Rate Limiting ─────────────────────────────────────────────
from app.routes.auth import limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── Register Routes ───────────────────────────────────────────
from app.routes import auth, persons, cameras, events, unknown_faces, analytics, movements, video_analysis, search, retention
from app.routes import browser_camera
from app.routes import share_camera_page
from app.routes import reid as reid_route

app.include_router(auth.router)
app.include_router(persons.router)
app.include_router(cameras.router)
app.include_router(events.router)
app.include_router(unknown_faces.router)
app.include_router(analytics.router)
app.include_router(movements.router)
app.include_router(video_analysis.router)
app.include_router(search.router)
app.include_router(retention.router)
app.include_router(browser_camera.router)
app.include_router(share_camera_page.router)
app.include_router(reid_route.router)


# ── Health Check ──────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    """System health check."""
    return {
        "status": "online",
        "system": "AI CCTV Surveillance System",
        "version": "1.0.0",
        "cameras": camera_manager.get_all_status(),
        "websocket_clients": ws_manager.event_count,
    }


@app.get("/health", tags=["Health"])
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "cameras": camera_manager.get_all_status(),
        "occupancy": camera_manager.get_all_occupancy(),
    }


# ── Global Error Handler ─────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )
