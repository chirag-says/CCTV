"""
In-memory store for mock/dev mode (no Supabase).

Stores detection events and unknown faces in memory
so the frontend can display them without a database.
"""

import threading
import logging
import pickle
import base64
from collections import OrderedDict
from typing import Optional, List
from datetime import datetime, timezone
from uuid import uuid4

logger = logging.getLogger(__name__)


class MockStore:
    """Thread-safe in-memory store for dev mode."""

    def __init__(self, max_unknown_faces: int = 100, max_events: int = 500):
        self._lock = threading.Lock()
        self._unknown_faces: OrderedDict[str, dict] = OrderedDict()
        self._events: list = []
        self._max_unknown = max_unknown_faces
        self._max_events = max_events
        self._persons: OrderedDict[str, dict] = OrderedDict()
        self._encodings: list = []
        self._cameras: OrderedDict[str, dict] = OrderedDict()

    # ── Cameras ───────────────────────────────────────────────

    def add_camera(self, data: dict) -> dict:
        """Store a camera."""
        cam_id = data.get("id", str(uuid4()))
        data["id"] = cam_id
        if "created_at" not in data:
            data["created_at"] = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._cameras[cam_id] = data
        logger.info(f"MockStore: Stored camera {cam_id} '{data.get('name')}' (total: {len(self._cameras)})")
        return data

    def list_cameras(self, is_active: Optional[bool] = None) -> list:
        """List all cameras."""
        with self._lock:
            cameras = list(self._cameras.values())
        if is_active is not None:
            cameras = [c for c in cameras if c.get("is_active", True) == is_active]
        cameras.sort(key=lambda c: c.get("created_at", ""), reverse=True)
        return cameras

    def get_camera(self, camera_id: str) -> Optional[dict]:
        """Get a camera by ID."""
        with self._lock:
            return self._cameras.get(camera_id)

    def update_camera(self, camera_id: str, updates: dict) -> Optional[dict]:
        """Update a camera."""
        with self._lock:
            if camera_id in self._cameras:
                self._cameras[camera_id].update(updates)
                return self._cameras[camera_id]
        return None

    def delete_camera(self, camera_id: str) -> bool:
        """Remove a camera."""
        with self._lock:
            return self._cameras.pop(camera_id, None) is not None

    # ── Persons ───────────────────────────────────────────────

    def add_person(self, data: dict) -> dict:
        """Store a person."""
        # Ensure ID
        if "id" not in data:
            data["id"] = str(uuid4())
        
        # Ensure timestamps
        now = datetime.now(timezone.utc).isoformat()
        if "created_at" not in data:
            data["created_at"] = now
        if "updated_at" not in data:
            data["updated_at"] = now
            
        pid = data["id"]
        with self._lock:
            self._persons[pid] = data
        return data

    def list_persons(
        self,
        role: Optional[str] = None,
        is_active: Optional[bool] = None,
        search: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        """List persons with filtering."""
        with self._lock:
            items = list(self._persons.values())

        # Filtering
        if role:
            items = [p for p in items if p.get("role") == role]
        if is_active is not None:
             items = [p for p in items if p.get("is_active", True) == is_active]
        if search:
            s_lower = search.lower()
            items = [p for p in items if s_lower in p.get("name", "").lower()]

        # Sort by created_at desc
        items.sort(key=lambda p: p.get("created_at", ""), reverse=True)

        total = len(items)
        data = items[offset : offset + limit]
        
        # Calculate encoding counts (inefficient but fine for mock)
        with self._lock:
            encs = self._encodings
            
        for p in data:
            p["encoding_count"] = sum(1 for e in encs if e.get("person_id") == p["id"])

        return {"data": data, "total": total}

    def get_person(self, person_id: str) -> Optional[dict]:
        with self._lock:
            p = self._persons.get(person_id)
        if p:
            # Add encoding count
            with self._lock:
                count = sum(1 for e in self._encodings if e.get("person_id") == person_id)
            p = p.copy()
            p["encoding_count"] = count
        return p

    def update_person(self, person_id: str, updates: dict) -> Optional[dict]:
        with self._lock:
            if person_id in self._persons:
                self._persons[person_id].update(updates)
                self._persons[person_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
                return self._persons[person_id]
        return None

    def delete_person(self, person_id: str) -> bool:
        with self._lock:
            if person_id in self._persons:
                self._persons[person_id]["is_active"] = False
                return True
        return False

    # ── Face Encodings ────────────────────────────────────────

    def add_encoding(self, data: dict) -> dict:
        if "id" not in data:
            data["id"] = str(uuid4())
        if "created_at" not in data:
            data["created_at"] = datetime.now(timezone.utc).isoformat()
            
        with self._lock:
            self._encodings.append(data)
        return data

    def get_encodings(self) -> list:
        with self._lock:
            # perform join for name
            results = []
            for enc in self._encodings:
                pid = enc.get("person_id")
                p = self._persons.get(pid)
                pname = p.get("name", "Unknown") if p else "Unknown"
                
                # Clone and add name
                item = enc.copy()
                item["person_name"] = pname
                results.append(item)
            return results

    # ── Unknown Faces ─────────────────────────────────────────

    def add_unknown_face(self, data: dict) -> dict:
        """Store an unknown face detection."""
        unknown_id = data.get("unknown_id", str(uuid4()))

        encoding = data.get("encoding")
        encoding_b64 = None
        if encoding is not None:
            try:
                encoding_b64 = base64.b64encode(pickle.dumps(encoding)).decode()
            except Exception:
                pass

        record = {
            "id": unknown_id,
            "camera_id": data.get("camera_id", ""),
            "snapshot_url": data.get("snapshot_path", ""),
            "context_url": data.get("context_path", ""),
            "full_frame": data.get("full_frame_path", ""),
            "encoding": encoding_b64,
            "occurrence": 1,
            "first_seen": data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "last_seen": data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        with self._lock:
            self._unknown_faces[unknown_id] = record
            # Evict oldest if over limit
            while len(self._unknown_faces) > self._max_unknown:
                self._unknown_faces.popitem(last=False)

        logger.info(f"MockStore: Stored unknown face {unknown_id} (total: {len(self._unknown_faces)})")
        return record

    def list_unknown_faces(
        self,
        status: Optional[str] = "pending",
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        """Get unknown faces, sorted by most recent."""
        with self._lock:
            faces = list(self._unknown_faces.values())

        # Filter by status
        if status:
            faces = [f for f in faces if f.get("status") == status]

        # Sort by last_seen descending
        faces.sort(key=lambda f: f.get("last_seen", ""), reverse=True)

        total = len(faces)
        data = faces[offset: offset + limit]

        return {"data": data, "total": total}

    def get_unknown_face(self, unknown_id: str) -> Optional[dict]:
        """Get a specific unknown face."""
        with self._lock:
            return self._unknown_faces.get(unknown_id)

    def update_unknown_face(self, unknown_id: str, updates: dict) -> Optional[dict]:
        """Update an unknown face record."""
        with self._lock:
            if unknown_id in self._unknown_faces:
                self._unknown_faces[unknown_id].update(updates)
                return self._unknown_faces[unknown_id]
        return None

    def remove_unknown_face(self, unknown_id: str) -> bool:
        """Remove an unknown face."""
        with self._lock:
            return self._unknown_faces.pop(unknown_id, None) is not None

    # ── Detection Events ──────────────────────────────────────

    def add_event(self, event: dict) -> dict:
        """Store a detection event."""
        # Ensure we store all relevant fields
        record = {
            "id": str(uuid4()),
            "person_id": event.get("person_id"),
            "camera_id": event.get("camera_id", ""),
            "event_type": event.get("event_type", "detection"),
            "subtype": event.get("subtype"),
            "person_name": event.get("person_name", "Unknown"),
            "confidence": event.get("confidence", 0.0),
            "snapshot_url": event.get("snapshot_url"),
            "timestamp": event.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": event.get("metadata", {}),
        }

        with self._lock:
            self._events.append(record)
            if len(self._events) > self._max_events:
                # Keep most recent
                self._events = self._events[-self._max_events:]

        return record

    def list_events(
        self,
        event_type: Optional[str] = None,
        subtype: Optional[str] = None,
        camera_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        """Get recent events with filtering."""
        with self._lock:
            # Events are stored oldest->newest, reverse for newest->oldest
            events = list(reversed(self._events))

        # Filter
        if event_type:
            events = [e for e in events if e.get("event_type") == event_type]
        if subtype:
            events = [e for e in events if e.get("subtype") == subtype]
        if camera_id:
            events = [e for e in events if e.get("camera_id") == camera_id]

        total = len(events)
        data = events[offset : offset + limit]
        return {"data": data, "total": total}


# Singleton instance
mock_store = MockStore()
