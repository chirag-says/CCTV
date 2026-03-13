"""
Unknown Faces API Routes — Admin Queue for reviewing and enrolling.
Now uses SQLAlchemy + PostgreSQL.
"""

import pickle
import logging
import numpy as np
from fastapi import APIRouter, Depends, Query, HTTPException
from typing import Optional

from app.models.event import UnknownFaceResponse, EnrollUnknownFace
from app.services.person_service import PersonService
from app.core.security import get_current_user, require_admin
from app.core.exceptions import NotFoundException
from app.db.session import SessionLocal
from app.db.models import UnknownFace as UnknownFaceDB, Person as PersonDB
from app.vision.camera_worker import camera_manager

from datetime import datetime, timezone
from uuid import uuid4

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/unknown-faces", tags=["Unknown Faces"])


def _unknown_to_dict(face: UnknownFaceDB) -> dict:
    return {
        "id": face.id,
        "camera_id": face.camera_id,
        "snapshot_url": face.snapshot_url,
        "full_frame": face.full_frame,
        "occurrence": face.occurrence,
        "first_seen": face.first_seen.isoformat() if face.first_seen else None,
        "last_seen": face.last_seen.isoformat() if face.last_seen else None,
        "status": face.status,
        "created_at": face.created_at.isoformat() if face.created_at else None,
    }


@router.get("")
async def list_unknown_faces(
    status: Optional[str] = Query("pending", description="pending/enrolled/dismissed"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_user),
):
    """List unknown faces queue, sorted by occurrence count."""
    db = SessionLocal()
    try:
        query = db.query(UnknownFaceDB)

        if status:
            query = query.filter(UnknownFaceDB.status == status)

        total = query.count()
        faces = (
            query.order_by(UnknownFaceDB.occurrence.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        return {
            "data": [_unknown_to_dict(f) for f in faces],
            "total": total,
        }
    finally:
        db.close()


@router.get("/{unknown_id}")
async def get_unknown_face(
    unknown_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get details for a specific unknown face entry."""
    db = SessionLocal()
    try:
        face = db.query(UnknownFaceDB).filter(UnknownFaceDB.id == unknown_id).first()
        if not face:
            raise NotFoundException("Unknown face", unknown_id)
        return _unknown_to_dict(face)
    finally:
        db.close()


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
    db = SessionLocal()
    try:
        # 1. Get the unknown face record
        face = db.query(UnknownFaceDB).filter(UnknownFaceDB.id == unknown_id).first()
        if not face:
            raise NotFoundException("Unknown face", unknown_id)

        if face.status != "pending":
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
        if face.face_encoding:
            try:
                encoding_array = pickle.loads(face.face_encoding)
                PersonService.add_face_encoding(
                    person_id=person["id"],
                    encoding=encoding_array,
                    source_image=face.snapshot_url,
                    quality=0.8,
                )
            except Exception as e:
                logger.warning(f"Failed to migrate encoding: {e}")

        # 4. Update person with avatar
        if face.snapshot_url:
            PersonService.update_person(person["id"], {"avatar_url": face.snapshot_url})

        # 5. Mark unknown face as enrolled
        face.status = "enrolled"
        db.commit()

        # 6. Reload encodings in all pipelines
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
    except (NotFoundException, HTTPException):
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Enrollment failed: {e}")
        raise HTTPException(status_code=500, detail="Enrollment failed")
    finally:
        db.close()


@router.post("/{unknown_id}/dismiss")
async def dismiss_unknown_face(
    unknown_id: str,
    current_user: dict = Depends(require_admin),
):
    """Dismiss an unknown face from the queue."""
    db = SessionLocal()
    try:
        face = db.query(UnknownFaceDB).filter(UnknownFaceDB.id == unknown_id).first()
        if not face:
            raise NotFoundException("Unknown face", unknown_id)

        face.status = "dismissed"
        db.commit()

        return {"message": f"Unknown face {unknown_id} dismissed"}
    except NotFoundException:
        raise
    except Exception as e:
        db.rollback()
        raise
    finally:
        db.close()
