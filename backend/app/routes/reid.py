"""
Person Re-Identification API Routes.

Exposes cross-camera person tracking data:
- List all persons seen across multiple cameras
- Get a specific person's cross-camera trail
- Get ReID system stats
"""

from fastapi import APIRouter, Depends, Query
from typing import Optional
from app.core.security import get_current_user
from app.vision.reid import get_reid_tracker

router = APIRouter(prefix="/api/reid", tags=["Re-Identification"])


@router.get("/persons")
async def get_reid_persons(
    cross_camera_only: bool = Query(False, description="Only show persons seen on 2+ cameras"),
    since_minutes: int = Query(30, description="Show persons seen in the last N minutes"),
    current_user: dict = Depends(get_current_user),
):
    """Get all tracked ReID persons."""
    tracker = get_reid_tracker()
    if cross_camera_only:
        persons = tracker.get_cross_camera_persons()
    else:
        persons = tracker.get_all_active(since_seconds=since_minutes * 60)

    return {
        "persons": persons,
        "total": len(persons),
        "stats": tracker.stats,
    }


@router.get("/persons/{reid_id}")
async def get_reid_person(reid_id: str, current_user: dict = Depends(get_current_user)):
    """Get a specific person's cross-camera trail."""
    tracker = get_reid_tracker()
    person = tracker.get_identity(reid_id)
    if not person:
        return {"error": "Person not found", "reid_id": reid_id}
    return person


@router.get("/persons/{reid_id}/trail")
async def get_reid_trail(reid_id: str, current_user: dict = Depends(get_current_user)):
    """Get just the camera trail for a person."""
    tracker = get_reid_tracker()
    person = tracker.get_identity(reid_id)
    if not person:
        return {"error": "Person not found", "reid_id": reid_id}
    return {
        "reid_id": reid_id,
        "person_name": person["person_name"],
        "trail": person["trail"],
        "cameras_seen": person["cameras_seen"],
    }


@router.get("/stats")
async def get_reid_stats(current_user: dict = Depends(get_current_user)):
    """Get ReID system statistics."""
    tracker = get_reid_tracker()
    return tracker.stats
