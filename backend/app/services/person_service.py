"""
Person Service — CRUD operations for persons and face encodings.
"""

import pickle
import logging
from typing import Optional, List, Dict, Any
from uuid import uuid4
from datetime import datetime, timezone

import numpy as np

from app.database import get_admin_db
from app.core.exceptions import NotFoundException
from app.core.mock_store import mock_store
from app.config import settings

logger = logging.getLogger(__name__)


class PersonService:
    """Service for person management operations."""

    @staticmethod
    def list_persons(
        role: Optional[str] = None,
        is_active: bool = True,
        search: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List persons with optional filtering."""
        db = get_admin_db()

        if db is None:
            return mock_store.list_persons(
                role=role,
                is_active=is_active,
                search=search,
                limit=limit,
                offset=offset,
            )

        query = db.table("persons").select("*", count="exact")

        if role:
            query = query.eq("role", role)
        if is_active is not None:
            query = query.eq("is_active", is_active)
        if search:
            query = query.ilike("name", f"%{search}%")

        query = query.order("created_at", desc=True)
        query = query.range(offset, offset + limit - 1)

        result = query.execute()

        # Get encoding counts for each person
        persons = result.data or []
        for person in persons:
            enc_result = (
                db.table("face_encodings")
                .select("id", count="exact")
                .eq("person_id", person["id"])
                .execute()
            )
            person["encoding_count"] = enc_result.count or 0

        return {
            "data": persons,
            "total": result.count or 0,
        }

    @staticmethod
    def get_person(person_id: str) -> dict:
        """Get a specific person by ID."""
        db = get_admin_db()

        if db is None:
            p = mock_store.get_person(person_id)
            if not p:
                raise NotFoundException("Person", person_id)
            return p

        result = db.table("persons").select("*").eq("id", person_id).execute()

        if not result.data:
            raise NotFoundException("Person", person_id)

        person = result.data[0]

        # Get encoding count
        enc_result = (
            db.table("face_encodings")
            .select("id", count="exact")
            .eq("person_id", person_id)
            .execute()
        )
        person["encoding_count"] = enc_result.count or 0

        return person

    @staticmethod
    def create_person(data: dict) -> dict:
        """Create a new person."""
        db = get_admin_db()

        person_data = {
            "id": str(uuid4()),
            "name": data["name"],
            "role": data.get("role", "visitor"),
            "department": data.get("department"),
            "phone": data.get("phone"),
            "email": data.get("email"),
            "is_active": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        if db is None:
            return mock_store.add_person(person_data)

        result = db.table("persons").insert(person_data).execute()
        if result.data:
            return result.data[0]

        raise Exception("Failed to create person")

    @staticmethod
    def update_person(person_id: str, data: dict) -> dict:
        """Update a person's information."""
        db = get_admin_db()

        if db is None:
            p = mock_store.update_person(person_id, data)
            if not p:
                raise NotFoundException("Person", person_id)
            return p

        # Filter out None values
        update_data = {k: v for k, v in data.items() if v is not None}
        update_data["updated_at"] = datetime.now(timezone.utc).isoformat()

        result = (
            db.table("persons")
            .update(update_data)
            .eq("id", person_id)
            .execute()
        )

        if not result.data:
            raise NotFoundException("Person", person_id)

        return result.data[0]

    @staticmethod
    def delete_person(person_id: str) -> bool:
        """Soft-delete a person."""
        db = get_admin_db()

        if db is None:
            if not mock_store.delete_person(person_id):
                 raise NotFoundException("Person", person_id)
            return True

        result = (
            db.table("persons")
            .update({"is_active": False, "updated_at": datetime.now(timezone.utc).isoformat()})
            .eq("id", person_id)
            .execute()
        )

        if not result.data:
            raise NotFoundException("Person", person_id)

        return True

    @staticmethod
    def add_face_encoding(
        person_id: str,
        encoding: np.ndarray,
        source_image: Optional[str] = None,
        quality: float = 1.0,
    ) -> dict:
        """Store a face encoding for a person."""
        db = get_admin_db()

        encoding_data = {
            "id": str(uuid4()),
            "person_id": person_id,
            "encoding": pickle.dumps(encoding),
            "source_image": source_image,
            "quality": quality,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        if db is None:
            mock_store.add_encoding(encoding_data)
            return {
                "id": encoding_data["id"],
                "person_id": person_id,
                "source_image": source_image,
                "quality": quality,
                "created_at": encoding_data["created_at"],
            }

        # Need to handle bytes for Supabase — store as base64 or use storage
        import base64
        db_data = encoding_data.copy()
        db_data["encoding"] = base64.b64encode(encoding_data["encoding"]).decode()

        result = db.table("face_encodings").insert(db_data).execute()

        if result.data:
            return {
                "id": result.data[0]["id"],
                "person_id": person_id,
                "source_image": source_image,
                "quality": quality,
                "created_at": result.data[0]["created_at"],
            }

        raise Exception("Failed to store face encoding")

    @staticmethod
    def get_all_encodings() -> list:
        """
        Load all face encodings from database.
        Returns list of dicts ready for FaceRecognizer.load_encodings().
        """
        db = get_admin_db()

        if db is None:
            raw_encs = mock_store.get_encodings()
            processed = []
            for item in raw_encs:
                try:
                    # item['encoding'] is pickled bytes
                    arr = pickle.loads(item['encoding'])
                    processed.append({
                        "encoding_id": item["id"],
                        "person_id": item["person_id"],
                        "person_name": item.get("person_name", "Unknown"),
                        "encoding": arr,
                        "quality": item.get("quality", 1.0),
                    })
                except Exception as e:
                    logger.warning(f"Mock encoding decode fail: {e}")
            return processed

        import base64

        # Join with persons to get names
        result = (
            db.table("face_encodings")
            .select("id, person_id, encoding, quality, persons(name)")
            .execute()
        )

        encodings = []
        for row in result.data or []:
            try:
                # Decode base64 → bytes → numpy array
                encoding_bytes = base64.b64decode(row["encoding"])
                encodings.append({
                    "encoding_id": row["id"],
                    "person_id": row["person_id"],
                    "person_name": row.get("persons", {}).get("name", "Unknown"),
                    "encoding": encoding_bytes,
                    "quality": row.get("quality", 1.0),
                })
            except Exception as e:
                logger.warning(f"Failed to decode encoding {row['id']}: {e}")

        logger.info(f"Loaded {len(encodings)} encodings from database")
        return encodings

    @staticmethod
    def get_person_history(
        person_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Get tracking session history for a person."""
        db = get_admin_db()

        if db is None:
            return {"data": [], "total": 0}

        result = (
            db.table("tracking_sessions")
            .select("*", count="exact")
            .eq("person_id", person_id)
            .order("entry_time", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )

        return {
            "data": result.data or [],
            "total": result.count or 0,
        }
