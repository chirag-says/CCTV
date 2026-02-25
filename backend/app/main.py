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

from app.config import settings
from app.services.person_service import PersonService
from app.vision.camera_worker import camera_manager
from app.core.websocket import ws_manager
from app.core.mock_store import mock_store
from app.utils.image_utils import frame_to_jpeg_bytes, resize_frame

# ── Logging Setup ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)-25s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cctv")

# Module-level reference to the main asyncio event loop (set during startup)
_main_loop: asyncio.AbstractEventLoop = None


# ── Event Handlers (Pipeline → WebSocket → DB) ───────────────

async def _broadcast_event(event: dict):
    """Forward pipeline event to WebSocket clients."""
    await ws_manager.broadcast_event(event)


async def _send_frame_async(camera_id: str, frame_bytes: bytes):
    """Send JPEG frame to WebSocket subscribers."""
    await ws_manager.send_frame(camera_id, frame_bytes)


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
    Bridges sync pipeline → async WebSocket.
    Also persists events to database.
    """
    logger.info(f"📡 Event: {event.get('event_type', 'unknown')} — {event.get('person_name', 'N/A')}")

    # Persist to database (async in background)
    try:
        from app.database import get_admin_db
        db = get_admin_db()
        if db:
            from uuid import uuid4
            from datetime import datetime, timezone

            event_data = {
                "id": str(uuid4()),
                "person_id": event.get("person_id"),
                "camera_id": event.get("camera_id", ""),
                "event_type": event.get("event_type", "detection"),
                "confidence": event.get("confidence", 0.0),
                "snapshot_url": event.get("snapshot_url"),
                "metadata": {
                    k: v for k, v in event.items()
                    if k not in ("person_id", "camera_id", "event_type", "confidence", "snapshot_url")
                },
                "created_at": event.get("timestamp", datetime.now(timezone.utc).isoformat()),
            }
            db.table("detection_events").insert(event_data).execute()

            # Create/update tracking session for entry/exit
            if event["event_type"] == "entry" and event.get("person_id"):
                session_data = {
                    "id": str(uuid4()),
                    "person_id": event["person_id"],
                    "camera_id": event["camera_id"],
                    "entry_time": event["timestamp"],
                    "status": "active",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                result = db.table("tracking_sessions").insert(session_data).execute()
                if result.data:
                    for cid in camera_manager.get_active_camera_ids():
                        worker = camera_manager._workers.get(cid)
                        if worker:
                            worker.pipeline.tracker.set_session_id(
                                event["person_id"], result.data[0]["id"]
                            )

            elif event["event_type"] == "exit" and event.get("session_id"):
                db.table("tracking_sessions").update({
                    "exit_time": event["timestamp"],
                    "duration_sec": event.get("duration_sec", 0),
                    "status": "completed",
                }).eq("id", event["session_id"]).execute()
        else:
            # Mock mode — store in memory
            mock_store.add_event(event)

    except Exception as e:
        logger.error(f"Failed to persist event: {e}")

    # Broadcast via WebSocket (fire-and-forget async)
    global _main_loop
    if _main_loop:
        try:
            asyncio.run_coroutine_threadsafe(_broadcast_event(event), _main_loop)
        except Exception:
            pass


def handle_unknown_face(data: dict):
    """
    Called when an unknown face is detected.
    Persists to unknown_faces table and broadcasts event.
    """
    logger.info(f"👤 Unknown face detected on camera {data.get('camera_id')}")

    try:
        from app.database import get_admin_db
        import pickle
        import base64

        db = get_admin_db()
        if db:
            from uuid import uuid4
            from datetime import datetime, timezone

            unknown_data = {
                "id": data.get("unknown_id", str(uuid4())),
                "camera_id": data.get("camera_id", ""),
                "snapshot_url": data.get("snapshot_path", ""),
                "context_url": data.get("context_path", ""),
                "full_frame": data.get("full_frame_path", ""),
                "encoding": base64.b64encode(
                    pickle.dumps(data["encoding"])
                ).decode() if "encoding" in data else None,
                "occurrence": 1,
                "first_seen": data.get("timestamp", datetime.now(timezone.utc).isoformat()),
                "last_seen": data.get("timestamp", datetime.now(timezone.utc).isoformat()),
                "status": "pending",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            db.table("unknown_faces").insert(unknown_data).execute()
        else:
            # Mock mode — store in memory
            mock_store.add_unknown_face(data)

    except Exception as e:
        logger.error(f"Failed to persist unknown face: {e}")

    # Broadcast
    handle_pipeline_event({
        "event_type": "unknown",
        "camera_id": data.get("camera_id", ""),
        "timestamp": data.get("timestamp", ""),
        "snapshot_url": data.get("snapshot_path", ""),
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
    logger.info("=" * 60)

    yield  # Application is running

    # Shutdown
    logger.info("🛑 Shutting down...")
    camera_manager.stop_all()
    logger.info("✅ All cameras stopped. Goodbye!")


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

# ── CORS ──────────────────────────────────────────────────────
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


# ── Register Routes ───────────────────────────────────────────
from app.routes import auth, persons, cameras, events, unknown_faces, analytics, movements, video_analysis

app.include_router(auth.router)
app.include_router(persons.router)
app.include_router(cameras.router)
app.include_router(events.router)
app.include_router(unknown_faces.router)
app.include_router(analytics.router)
app.include_router(movements.router)
app.include_router(video_analysis.router)


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
