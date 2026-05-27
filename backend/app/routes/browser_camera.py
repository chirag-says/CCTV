"""
Browser Camera Routes — Receive frames from remote browsers via WebSocket.

Allows any device with a browser to act as a remote camera by streaming
frames from getUserMedia to the backend for AI processing.
"""

import asyncio
import logging
import time
import threading
from typing import Dict, Optional

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/browser-cam", tags=["BrowserCamera"])

# ── Active browser camera sessions ──────────────────────────────
_browser_sessions: Dict[str, "BrowserCameraSession"] = {}


class BrowserCameraSession:
    """Holds state for a single browser-camera connection."""

    def __init__(self, session_id: str, label: str = "Remote Camera"):
        self.session_id = session_id
        self.label = label
        self.last_frame: Optional[np.ndarray] = None
        self.last_jpeg: Optional[bytes] = None
        self.connected_at = time.time()
        self.last_frame_at = 0.0
        self.frame_count = 0
        self.is_active = True

    def feed_frame(self, jpeg_bytes: bytes):
        """Decode a JPEG frame and store it."""
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            self.last_frame = frame
            self.last_jpeg = jpeg_bytes
            self.last_frame_at = time.time()
            self.frame_count += 1

    @property
    def status(self) -> dict:
        return {
            "session_id": self.session_id,
            "label": self.label,
            "is_active": self.is_active,
            "frame_count": self.frame_count,
            "connected_at": self.connected_at,
            "last_frame_at": self.last_frame_at,
            "fps": round(self.frame_count / max(1, time.time() - self.connected_at), 1),
        }


@router.get("/sessions")
async def list_sessions():
    """List all active browser camera sessions."""
    return {
        "sessions": [s.status for s in _browser_sessions.values() if s.is_active],
        "count": sum(1 for s in _browser_sessions.values() if s.is_active),
    }


@router.websocket("/stream/{session_id}")
async def browser_camera_stream(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for browser cameras.

    The browser sends JPEG frames as binary messages.
    The server decodes them and makes them available for viewing
    via the /feed endpoint and for AI processing.
    """
    await websocket.accept()

    # Get label from query params
    label = websocket.query_params.get("label", "Remote Camera")
    session = BrowserCameraSession(session_id=session_id, label=label)
    _browser_sessions[session_id] = session

    logger.info(f"Browser camera connected: {session_id} ({label})")

    from app.vision.camera_worker import camera_manager
    # Auto-start the Vision Pipeline for this browser session!
    camera_manager.start_camera(camera_id=session_id, stream_source=f"browser:{session_id}")

    try:
        while True:
            # Receive binary JPEG data from browser
            data = await websocket.receive_bytes()

            if len(data) < 100:  # Too small to be a valid JPEG
                continue

            session.feed_frame(data)

    except WebSocketDisconnect:
        logger.info(f"Browser camera disconnected: {session_id}")
    except Exception as e:
        logger.error(f"Browser camera error ({session_id}): {e}")
    finally:
        session.is_active = False
        camera_manager.stop_camera(session_id)
        # Keep session for a bit so viewers can see it went offline
        asyncio.get_event_loop().call_later(
            30, lambda sid=session_id: _browser_sessions.pop(sid, None)
        )


@router.get("/feed/{session_id}")
async def browser_camera_feed(session_id: str):
    """Get the latest JPEG frame from a browser camera."""
    from fastapi.responses import Response

    session = _browser_sessions.get(session_id)
    if not session or not session.last_jpeg:
        return Response(content=b"", status_code=204)

    return Response(
        content=session.last_jpeg,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Access-Control-Allow-Origin": "*",
        },
    )


@router.websocket("/view/{session_id}")
async def browser_camera_view(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for VIEWING a browser camera stream.
    Sends JPEG frames at ~15fps to any connected viewer.
    """
    await websocket.accept()

    session = _browser_sessions.get(session_id)
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return

    last_frame_id = None
    try:
        while session.is_active:
            if session.last_jpeg:
                current_id = id(session.last_jpeg)
                if current_id != last_frame_id:
                    last_frame_id = current_id
                    await websocket.send_bytes(session.last_jpeg)
            await asyncio.sleep(0.066)  # ~15fps
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
