"""
Zones — ground-plane calibration and live cross-camera state.

A zone is a set of cameras that share one floor coordinate frame (all the COE
corridor cameras, all the Front Gate cameras). Calibrating them is what lets
two cameras facing each other agree that they are looking at the same person:
see app/vision/ground_plane.py for why appearance cannot do that job here.

THE SHARED FRAME
----------------
Every camera in a zone is calibrated against THE SAME four physical floor
points, entered in the same order. That is the entire mechanism by which two
cameras end up in a common coordinate system — there is no other link between
them. The reference rectangle's real width and length turn those coordinates
into metres.

Image points are stored NORMALISED (0..1 fractions of frame width and height),
so a calibration survives a change of capture or stream resolution instead of
silently drifting.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from app.core.security import get_current_user
from app.services.camera_service import CameraService, camera_group
from app.vision.ground_plane import CalibrationError, GroundPlane
from app.vision.zone_tracker import get_zone_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/zones", tags=["zones"])


# ── Schemas ──────────────────────────────────────────────────────────────────

class CalibrationPayload(BaseModel):
    """Four (or more) floor correspondences for one camera."""

    zone: str = Field(..., min_length=1, max_length=128)
    image_points: List[List[float]] = Field(..., min_length=4)
    world_points: List[List[float]] = Field(..., min_length=4)

    @field_validator("image_points")
    @classmethod
    def _normalised(cls, v):
        for point in v:
            if len(point) != 2:
                raise ValueError("each image point must be [x, y]")
            # Clicks land inside the frame; a value outside 0..1 means the
            # caller sent pixels instead of fractions, which would produce a
            # plausible-looking homography that is wrong by the frame size.
            if not (-0.05 <= point[0] <= 1.05 and -0.05 <= point[1] <= 1.05):
                raise ValueError(
                    "image points must be normalised to 0..1 (fractions of "
                    "frame width and height), not pixels"
                )
        return v

    @field_validator("world_points")
    @classmethod
    def _pairs(cls, v):
        for point in v:
            if len(point) != 2:
                raise ValueError("each world point must be [x, y] in metres")
        return v


class CalibrationResponse(BaseModel):
    camera_id: str
    calibrated: bool
    zone: Optional[str] = None
    image_points: Optional[List[List[float]]] = None
    world_points: Optional[List[List[float]]] = None
    residual: Optional[float] = None
    updated_at: Optional[str] = None


# ── Zone listing ─────────────────────────────────────────────────────────────

@router.get("")
async def list_zones(current_user: dict = Depends(get_current_user)):
    """
    Every zone, its cameras, and how many are calibrated.

    Cross-camera tracking needs at least TWO calibrated cameras in a zone, so
    the count is what tells an operator whether a zone is actually working.
    """
    cameras = CameraService.list_cameras()
    zones: dict = {}
    for camera in cameras:
        name = camera.get("group") or camera_group(
            camera.get("name", ""), camera.get("location", "")
        )
        plane = GroundPlane.from_config(camera["id"], camera.get("config"))
        zones.setdefault(name, []).append({
            "id": camera["id"],
            "name": camera.get("name"),
            "status": camera.get("status"),
            "calibrated": plane is not None,
            "residual": round(plane.residual, 3) if plane else None,
            "calibrated_zone": plane.zone if plane else None,
        })

    out = []
    for name, members in sorted(zones.items()):
        calibrated = sum(1 for c in members if c["calibrated"])
        out.append({
            "zone": name,
            "cameras": members,
            "camera_count": len(members),
            "calibrated_count": calibrated,
            # One calibrated camera gives positions but nothing to cross-check
            # against; the feature only exists from two.
            "cross_camera_ready": calibrated >= 2,
        })
    return out


@router.get("/live")
async def zones_live(current_user: dict = Depends(get_current_user)):
    """Live cross-camera state for every zone with an active tracker."""
    registry = get_zone_registry()
    return {
        name: {"people": tracker.snapshot(), "stats": tracker.stats()}
        for name, tracker in registry.all_zones().items()
    }


@router.get("/{zone}/live")
async def zone_live(zone: str, current_user: dict = Depends(get_current_user)):
    """
    Who is currently in this zone, where, and which cameras can see them.

    `camera_count > 1` is the observable proof that cross-camera association is
    working: one human, several cameras, one position.
    """
    tracker = get_zone_registry().zone(zone)
    return {"zone": zone, "people": tracker.snapshot(), "stats": tracker.stats()}


# ── Per-camera calibration ───────────────────────────────────────────────────

@router.get("/cameras/{camera_id}/calibration", response_model=CalibrationResponse)
async def get_calibration(camera_id: str, current_user: dict = Depends(get_current_user)):
    camera = CameraService.get_camera(camera_id)
    plane = GroundPlane.from_config(camera_id, (camera or {}).get("config"))
    if plane is None:
        return CalibrationResponse(camera_id=camera_id, calibrated=False)
    return CalibrationResponse(
        camera_id=camera_id,
        calibrated=True,
        zone=plane.zone,
        image_points=[[float(u), float(v)] for u, v in plane.image_points],
        world_points=[[float(x), float(y)] for x, y in plane.world_points],
        residual=round(plane.residual, 4),
        updated_at=plane.updated_at,
    )


@router.put("/cameras/{camera_id}/calibration", response_model=CalibrationResponse)
async def save_calibration(
    camera_id: str,
    payload: CalibrationPayload,
    current_user: dict = Depends(get_current_user),
):
    """
    Store a camera's ground-plane calibration.

    Solved before saving, so a degenerate set of clicks is rejected here with a
    message an operator can act on rather than being written and then failing
    silently on the camera thread forever.
    """
    camera = CameraService.get_camera(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    if len(payload.image_points) != len(payload.world_points):
        raise HTTPException(
            status_code=400,
            detail=(
                f"point counts differ: {len(payload.image_points)} image vs "
                f"{len(payload.world_points)} world"
            ),
        )

    try:
        plane = GroundPlane.build(
            camera_id=camera_id,
            zone=payload.zone,
            image_points=payload.image_points,
            world_points=payload.world_points,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
    except CalibrationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    config = dict(camera.get("config") or {})
    config["ground_plane"] = plane.to_config()
    CameraService.update_camera(camera_id, {"config": config})

    logger.info(
        f"Ground-plane calibration saved for {camera_id} in zone {payload.zone!r} "
        f"(residual {plane.residual:.3f} m)"
    )
    return CalibrationResponse(
        camera_id=camera_id,
        calibrated=True,
        zone=plane.zone,
        image_points=[[float(u), float(v)] for u, v in plane.image_points],
        world_points=[[float(x), float(y)] for x, y in plane.world_points],
        residual=round(plane.residual, 4),
        updated_at=plane.updated_at,
    )


@router.delete("/cameras/{camera_id}/calibration")
async def delete_calibration(
    camera_id: str, current_user: dict = Depends(get_current_user)
):
    """Remove a calibration. The camera keeps working; it just leaves the zone."""
    camera = CameraService.get_camera(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    config = dict(camera.get("config") or {})
    if "ground_plane" not in config:
        return {"camera_id": camera_id, "calibrated": False, "message": "was not calibrated"}
    config.pop("ground_plane", None)
    CameraService.update_camera(camera_id, {"config": config})
    logger.info(f"Ground-plane calibration removed for {camera_id}")
    return {"camera_id": camera_id, "calibrated": False, "message": "calibration removed"}


@router.post("/cameras/{camera_id}/calibration/preview")
async def preview_calibration(
    camera_id: str,
    payload: CalibrationPayload,
    u: float = Query(..., description="normalised x of a pixel to test"),
    v: float = Query(..., description="normalised y of a pixel to test"),
    current_user: dict = Depends(get_current_user),
):
    """
    Where would a given pixel land on the floor, under these points?

    Lets the UI show an operator what their clicks actually mean before saving,
    which is the only way to catch a plausible-looking but wrong calibration.
    """
    try:
        plane = GroundPlane.build(
            camera_id=camera_id,
            zone=payload.zone,
            image_points=payload.image_points,
            world_points=payload.world_points,
        )
    except CalibrationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    position = plane.project(u, v)
    return {
        "residual": round(plane.residual, 4),
        "on_floor": position is not None,
        "x": round(position[0], 3) if position else None,
        "y": round(position[1], 3) if position else None,
    }
