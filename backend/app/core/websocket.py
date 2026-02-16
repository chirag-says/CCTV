"""
WebSocket connection manager for broadcasting live events.
"""

from fastapi import WebSocket
from typing import Dict, List, Set
import json
import logging
import asyncio

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for live event streaming."""

    def __init__(self):
        # General event subscribers
        self._event_connections: List[WebSocket] = []
        # Camera-specific stream subscribers: camera_id -> [websockets]
        self._stream_connections: Dict[str, List[WebSocket]] = {}
        self._lock = asyncio.Lock()

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
        """Accept a WebSocket connection for a camera stream."""
        await websocket.accept()
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


# Singleton instance
ws_manager = ConnectionManager()
