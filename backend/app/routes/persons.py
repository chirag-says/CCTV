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
    Runs quality gate to reject blurry/dark/angled faces.
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

    # Extract face location
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox
    face_location = (y1, x2, y2, x1)  # Convert to (top, right, bottom, left)

    # ── Face Quality Gate ─────────────────────────────────────
    from app.vision.quality import FaceQualityGate

    quality_gate = FaceQualityGate()

    # Extract face crop for quality assessment
    face_crop = image[max(0, y1):y2, max(0, x1):x2]

    # Get landmarks for pose estimation
    landmarks = face.landmark_2d_106 if hasattr(face, 'landmark_2d_106') else None
    if landmarks is None and hasattr(face, 'kps') and face.kps is not None:
        landmarks = face.kps  # 5-point landmarks

    quality_result = quality_gate.assess(
        face_crop=face_crop,
        face_location=face_location,
        det_score=float(face.det_score) if hasattr(face, 'det_score') else 1.0,
        landmarks=landmarks,
    )

    # Reject low-quality faces
    if not quality_result.passed:
        raise ValidationException(
            f"Face quality too low for enrollment. "
            f"Issues: {'; '.join(quality_result.rejection_reasons)}. "
            f"Quality score: {quality_result.quality_score:.2f}"
        )

    # ── Store encoding with metadata ──────────────────────────
    metrics = quality_result.metrics
    result = PersonService.add_face_encoding(
        person_id=person_id,
        encoding=encoding,
        source_image=file.filename,
        quality=quality_result.quality_score,
        yaw=metrics.get("yaw"),
        pitch=metrics.get("pitch"),
        roll=metrics.get("roll"),
        det_score=metrics.get("det_score"),
        blur_score=metrics.get("blur_score"),
        lighting=metrics.get("lighting"),
    )

    # Reload encodings in active pipelines
    all_encodings = PersonService.get_all_encodings()
    camera_manager.load_encodings_all(all_encodings)

    return {
        "message": "Face encoding created successfully",
        "encoding": result,
        "quality": {
            "score": quality_result.quality_score,
            "passed": True,
            "metrics": quality_result.metrics,
        },
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


# ── Encoding Management ─────────────────────────────────────────────

@router.get("/{person_id}/encodings/list")
async def list_person_encodings(
    person_id: str,
    current_user: dict = Depends(get_current_user),
):
    """List all face encodings for a person with metadata and health scores."""
    from app.services.embedding_suggestions import EmbeddingSuggestionService
    PersonService.get_person(person_id)
    return EmbeddingSuggestionService.get_person_encodings(person_id)


@router.delete("/{person_id}/encodings/{encoding_id}")
async def delete_encoding(
    person_id: str,
    encoding_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Delete a specific face encoding."""
    from app.services.embedding_suggestions import EmbeddingSuggestionService
    try:
        result = EmbeddingSuggestionService.delete_encoding(person_id, encoding_id)
        # Reload encodings in active pipelines
        all_encodings = PersonService.get_all_encodings()
        camera_manager.load_encodings_all(all_encodings)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── Embedding Suggestions ───────────────────────────────────────────

@router.get("/{person_id}/suggestions")
async def list_suggestions(
    person_id: str,
    status: str = Query("pending", description="Filter by status: pending/approved/rejected"),
    current_user: dict = Depends(get_current_user),
):
    """List suggested face encodings for a person."""
    from app.services.embedding_suggestions import EmbeddingSuggestionService
    PersonService.get_person(person_id)
    return EmbeddingSuggestionService.get_suggestions(person_id, status=status)


@router.get("/{person_id}/suggestions/count")
async def suggestion_count(
    person_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get count of pending suggestions for a person."""
    from app.services.embedding_suggestions import EmbeddingSuggestionService
    return {"count": EmbeddingSuggestionService.get_suggestion_count(person_id)}


@router.post("/{person_id}/suggestions/{suggestion_id}/approve")
async def approve_suggestion(
    person_id: str,
    suggestion_id: str,
    replace_encoding_id: Optional[str] = Query(None, description="ID of encoding to replace (if cap reached)"),
    current_user: dict = Depends(get_current_user),
):
    """
    Approve a suggested encoding.

    If the person has reached the encoding cap, returns replacement candidates
    unless replace_encoding_id is provided.
    """
    from app.services.embedding_suggestions import EmbeddingSuggestionService
    try:
        result = EmbeddingSuggestionService.approve_suggestion(
            suggestion_id=suggestion_id,
            replace_encoding_id=replace_encoding_id,
        )
        # Reload encodings in active pipelines if approved
        if result.get("status") == "approved":
            all_encodings = PersonService.get_all_encodings()
            camera_manager.load_encodings_all(all_encodings)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{person_id}/suggestions/{suggestion_id}/reject")
async def reject_suggestion(
    person_id: str,
    suggestion_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Reject a suggested encoding."""
    from app.services.embedding_suggestions import EmbeddingSuggestionService
    try:
        return EmbeddingSuggestionService.reject_suggestion(suggestion_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
