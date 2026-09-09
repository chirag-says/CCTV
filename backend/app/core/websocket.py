"""
WebSocket connection manager for broadcasting live events.
"""

from fastapi import WebSocket
from typing import Dict, List, Set
import json
import logging
import asyncio
import time

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for live event streaming."""

    def __init__(self):
        # General event subscribers
        self._event_connections: List[WebSocket] = []
        # Camera-specific stream subscribers: camera_id -> [websockets]
        self._stream_connections: Dict[str, List[WebSocket]] = {}
        # Camera-specific DETECTION METADATA subscribers: camera_id -> [ws].
        # Deliberately a separate table from _stream_connections. A metadata
        # subscriber is watching go2rtc's video, not ours, so it must NOT make
        # is_watched() true -- doing so would restart the JPEG encoder for
        # every camera on the monitor page, which is the cost Phase 3 removes.
        self._metadata_connections: Dict[str, List[WebSocket]] = {}
        self._lock = asyncio.Lock()
        # camera_id -> monotonic deadline until which a pull-based viewer
        # (the /feed snapshot endpoint) still counts as watching. A polling
        # <img> holds no connection, so it cannot be counted by presence;
        # it has to be counted by recency.
        self._snapshot_viewers: Dict[str, float] = {}

    async def connect_events(self, websocket: WebSocket):
        """Accept a WebSocket connection for live events."""
        await websocket.accept()
        async with self._lock:
            self._event_connections.append(websocket)
        logger.info(f"Event WebSocket connected. Total: {len(self._event_connections)}")

    async def disconnect_events(self, websocket: WebSocket):
        """Remove a WebSocket connection from events."""
        async with self._lock:
            if websocket in self._event_connections:
                self._event_connections.remove(websocket)
        logger.info(f"Event WebSocket disconnected. Total: {len(self._event_connections)}")

    async def connect_stream(self, camera_id: str, websocket: WebSocket):
        """Register an already-accepted WebSocket as a viewer of a camera.

        The caller owns accept(): the stream endpoint has to authenticate
        before accepting, so it cannot delegate that here.
        """
        async with self._lock:
            if camera_id not in self._stream_connections:
                self._stream_connections[camera_id] = []
            self._stream_connections[camera_id].append(websocket)
        logger.info(f"Stream WebSocket connected for camera {camera_id}")

    async def disconnect_stream(self, camera_id: str, websocket: WebSocket):
        """Remove a WebSocket from a camera stream."""
        async with self._lock:
            if camera_id in self._stream_connections:
                if websocket in self._stream_connections[camera_id]:
                    self._stream_connections[camera_id].remove(websocket)
                if not self._stream_connections[camera_id]:
                    del self._stream_connections[camera_id]

    # ── Detection metadata (Phase 3) ──────────────────────────────

    async def connect_metadata(self, camera_id: str, websocket: WebSocket):
        """Register an already-accepted socket as a metadata subscriber.

        As with connect_stream, the caller owns accept(): authentication has
        to happen before the handshake completes.
        """
        async with self._lock:
            self._metadata_connections.setdefault(camera_id, []).append(websocket)
        logger.info(f"Metadata WebSocket connected for camera {camera_id}")

    async def disconnect_metadata(self, camera_id: str, websocket: WebSocket):
        async with self._lock:
            conns = self._metadata_connections.get(camera_id)
            if not conns:
                return
            if websocket in conns:
                conns.remove(websocket)
            if not conns:
                del self._metadata_connections[camera_id]

    def has_metadata_subscribers(self, camera_id: str) -> bool:
        """Read from worker threads without a lock, like is_watched(): the
        pipeline asks this to decide whether building a message is worth it,
        and a one-frame-stale answer costs nothing."""
        return bool(self._metadata_connections.get(camera_id))

    async def broadcast_metadata(self, camera_id: str, message: dict):
        """Send one frame's detections to that camera's subscribers.

        JSON text, never bytes. Nothing on this channel is an image.
        """
        conns = self._metadata_connections.get(camera_id)
        if not conns:
            return

        payload = json.dumps(message)
        disconnected = []
        for connection in list(conns):
            try:
                await connection.send_text(payload)
            except Exception:
                disconnected.append(connection)

        for conn in disconnected:
            await self.disconnect_metadata(camera_id, conn)

    @property
    def metadata_counts(self) -> Dict[str, int]:
        return {k: len(v) for k, v in self._metadata_connections.items()}

    async def broadcast_event(self, event_data: dict):
        """Broadcast an event to all connected event subscribers."""
        if not self._event_connections:
            return

        message = json.dumps(event_data)
        disconnected = []

        for connection in self._event_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)

        # Clean up disconnected
        for conn in disconnected:
            await self.disconnect_events(conn)

    async def send_frame(self, camera_id: str, frame_data: bytes):
        """Send a video frame to all subscribers of a camera."""
        if camera_id not in self._stream_connections:
            return

        disconnected = []
        for connection in self._stream_connections[camera_id]:
            try:
                await connection.send_bytes(frame_data)
            except Exception:
                disconnected.append(connection)

        for conn in disconnected:
            await self.disconnect_stream(camera_id, conn)

    @property
    def event_count(self) -> int:
        return len(self._event_connections)

    @property
    def stream_counts(self) -> Dict[str, int]:
        return {k: len(v) for k, v in self._stream_connections.items()}

    # ── Viewer presence ───────────────────────────────────────────
    # Read from camera worker threads, not from the event loop, so these
    # take no lock. Both structures are plain dicts mutated by whole-key
    # assignment, which is atomic under the GIL; a viewer that appears or
    # vanishes one frame early is not worth a lock on the hot path.

    def note_snapshot_request(self, camera_id: str, ttl_sec: float = 2.0):
        """Record that someone pulled a still of this camera.

        Called by the /feed endpoint. The TTL has to outlast the client's
        polling interval, or a steadily-polling viewer would look idle
        between requests and the encoder would stop underneath it.
        """
        self._snapshot_viewers[camera_id] = time.monotonic() + ttl_sec

    def is_watched(self, camera_id: str) -> bool:
        """True if anything is currently consuming this camera's video."""
        if self._stream_connections.get(camera_id):
            return True
        deadline = self._snapshot_viewers.get(camera_id)
        return deadline is not None and deadline > time.monotonic()

    def viewer_count(self, camera_id: str) -> int:
        n = len(self._stream_connections.get(camera_id) or ())
        deadline = self._snapshot_viewers.get(camera_id)
        if deadline is not None and deadline > time.monotonic():
            n += 1
        return n


# Singleton instance
ws_manager = ConnectionManager()
