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
from app.core.exceptions import ValidationException, NotFoundException
from app.vision.camera_worker import camera_manager

router = APIRouter(prefix="/api/persons", tags=["Persons"])


@router.get("")
async def list_persons(
    role: Optional[str] = Query(None, description="Filter by designation (assistant_professor, student, …)"),
    person_type: Optional[str] = Query(None, description="Filter by class: faculty/student/staff/visitor"),
    department: Optional[str] = Query(None, description="Filter by department (case-insensitive)"),
    is_active: bool = Query(True, description="Filter by active status"),
    search: Optional[str] = Query(None, description="Search by name"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_user),
):
    """List all known persons with optional filtering."""
    result = PersonService.list_persons(
        role=role,
        person_type=person_type,
        department=department,
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
    hard: bool = Query(False, description="Permanently delete person and all data (irreversible)"),
    current_user: dict = Depends(get_current_user),
):
    """
    Delete a person.

    - Default (hard=false): Soft-delete — sets is_active=False, person remains in DB.
    - hard=true: Permanently removes person, all encodings, suggestions,
      events, and tracking sessions. Reloads recognizer. IRREVERSIBLE.
    """
    if hard:
        from app.db.session import SessionLocal
        from app.db.models import (
            Person, FaceEncoding, SuggestedEncoding,
            DetectionEvent, TrackingSession,
        )

        db = SessionLocal()
        try:
            person = db.query(Person).filter(Person.id == person_id).first()
            if not person:
                raise HTTPException(status_code=404, detail=f"Person {person_id} not found")

            person_name = person.name

            # Delete all related data (order matters due to FK constraints)
            db.query(SuggestedEncoding).filter(
                SuggestedEncoding.person_id == person_id
            ).delete(synchronize_session=False)

            db.query(FaceEncoding).filter(
                FaceEncoding.person_id == person_id
            ).delete(synchronize_session=False)

            db.query(DetectionEvent).filter(
                DetectionEvent.person_id == person_id
            ).delete(synchronize_session=False)

            db.query(TrackingSession).filter(
                TrackingSession.person_id == person_id
            ).delete(synchronize_session=False)

            db.delete(person)

            # Bump encoding version so pipelines reload
            from app.services.person_service import PersonService as PS
            current_version = PS._get_or_create_version(db)
            PS._set_version(db, current_version + 1)

            db.commit()

            # Reload recognizer cache immediately
            try:
                all_encodings = PersonService.get_all_encodings()
                camera_manager.load_encodings_all(all_encodings)
            except Exception:
                pass

            # Remove from event queue FK guard
            try:
                from app.core.event_queue import get_event_queue
                get_event_queue().remove_person_id(person_id)
            except Exception:
                pass

            return {
                "message": f"Person '{person_name}' permanently deleted",
                "person_id": person_id,
                "hard_delete": True,
            }
        except HTTPException:
            raise
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Hard delete failed: {str(e)}")
        finally:
            db.close()
    else:
        PersonService.delete_person(person_id)
        return {"message": f"Person {person_id} deactivated", "person_id": person_id, "hard_delete": False}


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

    # Pass BGR directly — InsightFace expects BGR input and handles
    # BGR→RGB internally via cv2.dnn.blobFromImage(swapRB=True).
    # Do NOT convert to RGB: that double-swaps the channels.
    from app.vision.detector import _get_insightface_app

    app = _get_insightface_app()
    faces = app.get(image)  # BGR input — correct for InsightFace

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


@router.get("/{person_id}/journey")
async def get_person_journey(
    person_id: str,
    date: Optional[str] = Query(
        None, description="Campus-local day (YYYY-MM-DD). Defaults to today."
    ),
    current_user: dict = Depends(get_current_user),
):
    """
    Where this person was seen on a given day, in order.

    Consecutive detections at the same physical camera are collapsed into one
    "stop", so a person lingering at the gate appears once rather than five
    times. Returns the stop sequence, the gaps between them, and day totals.

    Times are first/last DETECTION, not gate crossings — a person out of
    camera view is simply not observed.
    """
    from app.services.journey_service import JourneyService

    try:
        return JourneyService.get_journey(person_id=person_id, date=date)
    except NotFoundException:
        raise HTTPException(status_code=404, detail=f"Person {person_id} not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


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
