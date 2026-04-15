"""
Persons Management API Routes.
"""

import cv2
import numpy as np
import tempfile
import os
from fastapi import APIRouter, Depends, Query, UploadFile, File, HTTPException
from typing import Optional, List

from app.models.person import PersonCreate, PersonUpdate, PersonResponse, PersonWithHistory
from app.services.person_service import PersonService
from app.core.security import get_current_user
from app.core.exceptions import ValidationException
from app.vision.camera_worker import camera_manager

router = APIRouter(prefix="/api/persons", tags=["Persons"])


@router.get("")
async def list_persons(
    role: Optional[str] = Query(None, description="Filter by role"),
    is_active: bool = Query(True, description="Filter by active status"),
    search: Optional[str] = Query(None, description="Search by name"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_user),
):
    """List all known persons with optional filtering."""
    result = PersonService.list_persons(
        role=role,
        is_active=is_active,
        search=search,
        limit=limit,
        offset=offset,
    )
    return result


@router.post("", response_model=PersonResponse)
async def create_person(
    data: PersonCreate,
    current_user: dict = Depends(get_current_user),
):
    """Add a new known person."""
    result = PersonService.create_person(data.model_dump())
    return result


@router.get("/{person_id}")
async def get_person(
    person_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get details for a specific person."""
    return PersonService.get_person(person_id)


@router.put("/{person_id}", response_model=PersonResponse)
async def update_person(
    person_id: str,
    data: PersonUpdate,
    current_user: dict = Depends(get_current_user),
):
    """Update a person's information."""
    return PersonService.update_person(person_id, data.model_dump(exclude_unset=True))


@router.delete("/{person_id}")
async def delete_person(
    person_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Soft-delete a person."""
    PersonService.delete_person(person_id)
    return {"message": f"Person {person_id} deleted"}


@router.post("/{person_id}/encodings")
async def upload_face_encoding(
    person_id: str,
    file: UploadFile = File(..., description="Face image (JPEG/PNG)"),
    current_user: dict = Depends(get_current_user),
):
    """
    Upload a face image for a person.
    Generates ArcFace embedding (512-d) using SCRFD detection + insightface.
    """
    # Validate person exists
    PersonService.get_person(person_id)

    # Validate file type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise ValidationException("Only JPEG and PNG images are supported")

    # Read and decode image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValidationException("Failed to decode image")

    # Convert to RGB for insightface
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Use insightface (SCRFD + ArcFace) for detection and embedding
    from app.vision.detector import _get_insightface_app

    app = _get_insightface_app()
    faces = app.get(rgb_image)

    if not faces:
        raise ValidationException("No face detected in the uploaded image")

    if len(faces) > 1:
        raise ValidationException(
            f"Multiple faces ({len(faces)}) detected. "
            "Please upload an image with exactly one face."
        )

    face = faces[0]

    # Extract ArcFace embedding (512-d)
    if face.embedding is None:
        raise ValidationException("Failed to generate face embedding")

    encoding = face.embedding  # 512-d numpy array

    # Extract face location for quality assessment
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox
    face_location = (y1, x2, y2, x1)  # Convert to (top, right, bottom, left)

    # Assess quality
    from app.vision.detector import FaceDetector
    detector = FaceDetector()
    quality = detector.assess_face_quality(image, face_location)

    # Store encoding
    result = PersonService.add_face_encoding(
        person_id=person_id,
        encoding=encoding,
        source_image=file.filename,
        quality=quality,
    )

    # Reload encodings in active pipelines
    all_encodings = PersonService.get_all_encodings()
    camera_manager.load_encodings_all(all_encodings)

    return {
        "message": "Face encoding created successfully",
        "encoding": result,
        "quality": quality,
        "face_location": {
            "top": face_location[0],
            "right": face_location[1],
            "bottom": face_location[2],
            "left": face_location[3],
        },
    }


@router.get("/{person_id}/history")
async def get_person_history(
    person_id: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_user),
):
    """Get tracking session history for a person."""
    # Validate person exists
    PersonService.get_person(person_id)
    return PersonService.get_person_history(person_id, limit, offset)
