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
from app.vision.handoff import MAX_GAP_SEC, get_handoff, reload_handoff_config, topology_path
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


class HandoffAreaModel(BaseModel):
    """A region of one camera's frame where people leave or enter its view."""

    id: str = Field(..., min_length=1, max_length=64)
    name: str = Field(..., min_length=1, max_length=64)
    rect: List[float] = Field(..., min_length=4, max_length=4)

    @field_validator("rect")
    @classmethod
    def _normalised_rect(cls, v):
        x1, y1, x2, y2 = v
        # Same trap as calibration points: pixels would validate as a huge
        # rectangle that contains every foot point in the frame.
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise ValueError("rect must be [x1, y1, x2, y2] normalised to 0..1 with x1<x2, y1<y2")
        if (x2 - x1) < 0.02 or (y2 - y1) < 0.02:
            raise ValueError("rect is too small to contain a person's feet")
        return v


class HandoffCameraModel(BaseModel):
    id: str
    areas: List[HandoffAreaModel] = Field(default_factory=list, max_length=8)


class AreaRefModel(BaseModel):
    camera_id: str
    area_id: str


class HandoffLinkModel(BaseModel):
    """Two areas a person passes between, at most max_gap_sec apart. Undirected."""

    a: AreaRefModel
    b: AreaRefModel
    max_gap_sec: float = Field(..., gt=0, le=MAX_GAP_SEC)


class HandoffDocument(BaseModel):
    cameras: List[HandoffCameraModel] = Field(default_factory=list)
    links: List[HandoffLinkModel] = Field(default_factory=list, max_length=64)


class CalibrationResponse(BaseModel):
    camera_id: str
    calibrated: bool
    zone: Optional[str] = None
    image_points: Optional[List[List[float]]] = None
    world_points: Optional[List[List[float]]] = None
    residual: Optional[float] = None
    updated_at: Optional[str] = None
    # Non-empty when the calibration is stored but cannot locate people. The
    # engine ignores such a calibration; the operator has to redo the clicks.
    problems: List[str] = Field(default_factory=list)


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
        problems = plane.problems() if plane else []
        zones.setdefault(name, []).append({
            "id": camera["id"],
            "name": camera.get("name"),
            "status": camera.get("status"),
            "calibrated": plane is not None,
            # Saved is not the same as usable: four clicks always fit
            # themselves, so the residual cannot tell a floor rectangle from
            # four points up a door post. `usable` can, and is what the
            # engine actually goes by.
            "usable": plane is not None and not problems,
            "problems": problems,
            "residual": round(plane.residual, 3) if plane else None,
            "calibrated_zone": plane.zone if plane else None,
        })

    out = []
    for name, members in sorted(zones.items()):
        calibrated = sum(1 for c in members if c["usable"])
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


# ── Cross-camera handoff areas and links ─────────────────────────────────────

def _zone_cameras(zone: str) -> List[dict]:
    """Cameras in a zone, grouped exactly as list_zones groups them."""
    return [
        c for c in CameraService.list_cameras()
        if (c.get("group") or camera_group(c.get("name", ""), c.get("location", ""))) == zone
    ]


@router.get("/{zone}/handoff")
async def get_handoff_config(zone: str, current_user: dict = Depends(get_current_user)):
    """
    Handoff areas for every camera in the zone, and the links between them.

    Links are returned zone-wide, in one list, because a link has two ends on
    two cameras and editing it from either camera's page alone is how the two
    halves drift apart.
    """
    cameras = _zone_cameras(zone)
    out_cameras, links = [], []
    for camera in cameras:
        handoff = (camera.get("config") or {}).get("handoff") or {}
        out_cameras.append({
            "id": camera["id"],
            "name": camera.get("name"),
            "areas": handoff.get("areas") or [],
        })
        for link in handoff.get("links") or []:
            links.append({
                "a": {"camera_id": camera["id"], "area_id": link.get("area_id")},
                "b": {"camera_id": link.get("to_camera_id"), "area_id": link.get("to_area_id")},
                "max_gap_sec": link.get("max_gap_sec"),
            })
    return {"zone": zone, "cameras": out_cameras, "links": links, "max_gap_sec": MAX_GAP_SEC}


@router.put("/{zone}/handoff")
async def save_handoff_config(
    zone: str,
    payload: HandoffDocument,
    current_user: dict = Depends(get_current_user),
):
    """
    Replace the zone's handoff areas and links in one write.

    The whole zone document is validated before anything is written, so a
    link to a deleted area is rejected here instead of silently becoming a
    link the engine skips forever.
    """
    cameras = {c["id"]: c for c in _zone_cameras(zone)}
    if not cameras:
        raise HTTPException(status_code=404, detail=f"No cameras in zone {zone!r}")

    areas_by_camera = {cam_id: [] for cam_id in cameras}
    for entry in payload.cameras:
        if entry.id not in cameras:
            raise HTTPException(status_code=400, detail=f"Camera {entry.id} is not in zone {zone!r}")
        ids = [a.id for a in entry.areas]
        if len(ids) != len(set(ids)):
            raise HTTPException(status_code=400, detail=f"Duplicate area id on camera {entry.id}")
        areas_by_camera[entry.id] = entry.areas

    known = {(cam, a.id) for cam, areas in areas_by_camera.items() for a in areas}
    links_by_camera = {cam_id: [] for cam_id in cameras}
    seen = set()
    for link in payload.links:
        a = (link.a.camera_id, link.a.area_id)
        b = (link.b.camera_id, link.b.area_id)
        if a not in known or b not in known:
            raise HTTPException(status_code=400, detail="A link refers to an area that does not exist")
        if a[0] == b[0]:
            raise HTTPException(
                status_code=400,
                detail="A link must join areas on two different cameras",
            )
        pair = frozenset((a, b))
        if pair in seen:
            continue
        seen.add(pair)
        links_by_camera[a[0]].append({
            "area_id": a[1],
            "to_camera_id": b[0],
            "to_area_id": b[1],
            "max_gap_sec": float(link.max_gap_sec),
        })

    for cam_id, camera in cameras.items():
        config = dict(camera.get("config") or {})
        areas = [a.model_dump() for a in areas_by_camera[cam_id]]
        links = links_by_camera[cam_id]
        if areas or links:
            config["handoff"] = {"areas": areas, "links": links}
        elif "handoff" in config:
            config.pop("handoff")
        else:
            continue
        CameraService.update_camera(cam_id, {"config": config})

    reload_handoff_config()
    logger.info(
        f"Handoff config saved for zone {zone!r}: "
        f"{len(known)} areas, {len(seen)} links"
    )
    return await get_handoff_config(zone, current_user)


@router.get("/{zone}/handoff/live")
async def handoff_live(zone: str, current_user: dict = Depends(get_current_user)):
    """
    Global numbers currently held by people seen in this zone, and the path
    each took. A path with more than one camera is the observable proof that
    handoff is working.
    """
    engine = get_handoff()
    if engine is None:
        return {"zone": zone, "enabled": False, "people": [], "stats": {}}
    snapshot = engine.snapshot(c["id"] for c in _zone_cameras(zone))
    return {"zone": zone, "enabled": True, **snapshot}


@router.delete("/{zone}/handoff/learned")
async def forget_learned(zone: str, current_user: dict = Depends(get_current_user)):
    """
    Forget what handoff has learned about this zone's cameras: which views
    overlap, and the routes people walk between them.

    For after a camera is moved or re-aimed. It is not required -- stale
    samples age out as new traffic disagrees with them -- but it gets there in
    an hour of ordinary traffic instead of a day. Nothing an operator entered
    (calibrations, areas, links) is touched.
    """
    engine = get_handoff()
    if engine is None:
        raise HTTPException(status_code=409, detail="Handoff is switched off (HANDOFF_ENABLED=false)")
    cameras = _zone_cameras(zone)
    if not cameras:
        raise HTTPException(status_code=404, detail=f"No cameras in zone {zone!r}")
    dropped = engine.forget_learned(c["id"] for c in cameras)
    engine.save_topology(topology_path(), force=True)
    logger.info(f"Learned handoff topology for zone {zone!r} cleared ({dropped} relation(s))")
    return {"zone": zone, "forgotten": dropped}


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
        problems=plane.problems(),
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

    # Refuse a calibration that solves but cannot locate anyone. Saving it
    # would show "fit error 0 cm" and then silently never associate -- which
    # is exactly how both COE corridor cameras sat broken.
    problems = plane.problems()
    if problems:
        raise HTTPException(
            status_code=400,
            detail="This calibration cannot be used: " + " Also: ".join(problems),
        )

    config = dict(camera.get("config") or {})
    config["ground_plane"] = plane.to_config()
    CameraService.update_camera(camera_id, {"config": config})
    reload_handoff_config()

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
    reload_handoff_config()
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
        "problems": plane.problems(),
        "residual": round(plane.residual, 4),
        "on_floor": position is not None,
        "x": round(position[0], 3) if position else None,
        "y": round(position[1], 3) if position else None,
    }
