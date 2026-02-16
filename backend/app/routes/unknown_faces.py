"""
Unknown Faces API Routes — Admin Queue for reviewing and enrolling.
"""

import pickle
import base64
import logging
import numpy as np
from fastapi import APIRouter, Depends, Query
from typing import Optional

from app.models.event import UnknownFaceResponse, EnrollUnknownFace
from app.services.person_service import PersonService
from app.core.security import get_current_user, require_admin
from app.core.exceptions import NotFoundException
from app.core.mock_store import mock_store
from app.database import get_admin_db
from app.vision.camera_worker import camera_manager
from app.vision.recognizer import KnownFace

from datetime import datetime, timezone
from uuid import uuid4

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/unknown-faces", tags=["Unknown Faces"])


@router.get("")
async def list_unknown_faces(
    status: Optional[str] = Query("pending", description="pending/enrolled/dismissed"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_user),
):
    """List unknown faces queue, sorted by occurrence count."""
    db = get_admin_db()

    if db is None:
        # Mock mode — use in-memory store
        return mock_store.list_unknown_faces(status=status, limit=limit, offset=offset)

    query = db.table("unknown_faces").select("*", count="exact")

    if status:
        query = query.eq("status", status)

    query = query.order("occurrence", desc=True)
    query = query.range(offset, offset + limit - 1)

    result = query.execute()

    return {
        "data": result.data or [],
        "total": result.count or 0,
    }


@router.get("/{unknown_id}")
async def get_unknown_face(
    unknown_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get details for a specific unknown face entry."""
    db = get_admin_db()

    if db is None:
        # Mock mode — use in-memory store
        face = mock_store.get_unknown_face(unknown_id)
        if not face:
            raise NotFoundException("Unknown face", unknown_id)
        return face

    result = db.table("unknown_faces").select("*").eq("id", unknown_id).execute()

    if not result.data:
        raise NotFoundException("Unknown face", unknown_id)

    return result.data[0]


@router.post("/{unknown_id}/enroll")
async def enroll_unknown_face(
    unknown_id: str,
    data: EnrollUnknownFace,
    current_user: dict = Depends(require_admin),
):
    """
    Enroll an unknown face as a known person.

    Steps:
    1. Create new person record
    2. Move face encoding from unknown_faces to face_encodings
    3. Mark unknown_face as enrolled
    4. Reload encoding cache in all active pipelines
    """
    db = get_admin_db()

    if db is None:
        # Mock mode — enroll from in-memory store
        face = mock_store.get_unknown_face(unknown_id)
        if not face:
            raise NotFoundException("Unknown face", unknown_id)

        # 1. Create person
        person = PersonService.create_person({
            "name": data.name,
            "role": data.role,
            "department": data.department,
            "phone": data.phone,
            "email": data.email,
        })

        # 2. Add encoding
        if face.get("encoding"):
            try:
                # face["encoding"] is base64(pickle(numpy_array))
                encoding_bytes = base64.b64decode(face["encoding"])
                encoding_array = pickle.loads(encoding_bytes)

                PersonService.add_face_encoding(
                    person_id=person["id"],
                    encoding=encoding_array,
                    source_image=face.get("snapshot_url"),
                    quality=0.8,
                )
            except Exception as e:
                logger.warning(f"Failed to migrate encoding: {e}")

        # 3. Update avatar if available
        if face.get("snapshot_url"):
            PersonService.update_person(person["id"], {"avatar_url": face["snapshot_url"]})

        # 4. Mark unknown face as enrolled
        mock_store.update_unknown_face(unknown_id, {"status": "enrolled"})

        # 5. Reload encodings in all pipelines
        try:
            all_encodings = PersonService.get_all_encodings()
            camera_manager.load_encodings_all(all_encodings)
        except Exception as e:
            logger.error(f"Failed to reload encodings: {e}")

        return {
            "message": f"Successfully enrolled {data.name}",
            "person_id": person["id"],
            "person": person,
        }

    # 1. Get the unknown face record
    unknown_result = db.table("unknown_faces").select("*").eq("id", unknown_id).execute()
    if not unknown_result.data:
        raise NotFoundException("Unknown face", unknown_id)

    unknown_face = unknown_result.data[0]

    if unknown_face["status"] != "pending":
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Unknown face already processed")

    # 2. Create person
    person = PersonService.create_person({
        "name": data.name,
        "role": data.role,
        "department": data.department,
        "phone": data.phone,
        "email": data.email,
    })

    # 3. Move encoding
    if unknown_face.get("encoding"):
        try:
            encoding_bytes = base64.b64decode(unknown_face["encoding"])
            encoding_array = pickle.loads(encoding_bytes)

            PersonService.add_face_encoding(
                person_id=person["id"],
                encoding=encoding_array,
                source_image=unknown_face.get("snapshot_url"),
                quality=0.8,
            )
        except Exception as e:
            logger.warning(f"Failed to migrate encoding: {e}")

    # 4. Update person with avatar
    if unknown_face.get("snapshot_url"):
        db.table("persons").update({
            "avatar_url": unknown_face["snapshot_url"]
        }).eq("id", person["id"]).execute()

    # 5. Mark unknown face as enrolled
    db.table("unknown_faces").update({
        "status": "enrolled",
        "assigned_to": current_user.get("id"),
    }).eq("id", unknown_id).execute()

    # 6. Reload encodings in all pipelines
    all_encodings = PersonService.get_all_encodings()
    camera_manager.load_encodings_all(all_encodings)

    return {
        "message": f"Successfully enrolled {data.name}",
        "person_id": person["id"],
        "person": person,
    }


@router.post("/{unknown_id}/dismiss")
async def dismiss_unknown_face(
    unknown_id: str,
    current_user: dict = Depends(require_admin),
):
    """Dismiss an unknown face from the queue."""
    db = get_admin_db()

    if db is None:
        # Mock mode — dismiss from in-memory store
        face = mock_store.get_unknown_face(unknown_id)
        if not face:
            raise NotFoundException("Unknown face", unknown_id)
        mock_store.update_unknown_face(unknown_id, {"status": "dismissed"})
        return {"message": f"Unknown face {unknown_id} dismissed"}

    result = (
        db.table("unknown_faces")
        .update({
            "status": "dismissed",
            "assigned_to": current_user.get("id"),
        })
        .eq("id", unknown_id)
        .execute()
    )

    if not result.data:
        raise NotFoundException("Unknown face", unknown_id)

    return {"message": f"Unknown face {unknown_id} dismissed"}
