"""
Camera Management API Routes.
"""

import asyncio
import logging

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import StreamingResponse
from typing import Optional

from app.models.camera import CameraCreate, CameraUpdate, CameraResponse
from app.services.camera_service import CameraService
from app.core.security import get_current_user
from app.core.websocket import ws_manager
from app.vision.camera_worker import camera_manager


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cameras", tags=["Cameras"])


@router.get("/active-persons")
async def get_active_persons():
    """
    Get all currently detected/present persons across all cameras.
    Returns real-time tracking data from all active pipelines.
    """
    all_persons = []
    seen_ids = set()

    for camera_id in camera_manager.get_active_camera_ids():
        worker = camera_manager._workers.get(camera_id)
        if worker and worker.pipeline:
            for person in worker.pipeline.tracker.get_active_persons():
                pid = person["person_id"]
                if pid not in seen_ids:
                    seen_ids.add(pid)
                    all_persons.append(person)

    return {
        "persons": all_persons,
        "count": len(all_persons),
    }


@router.get("")
async def list_cameras(
    is_active: Optional[bool] = Query(None),
    current_user: dict = Depends(get_current_user),
):
    """List all registered cameras with status."""
    return CameraService.list_cameras(is_active=is_active)


@router.post("", response_model=CameraResponse)
async def create_camera(
    data: CameraCreate,
    current_user: dict = Depends(get_current_user),
):
    """Register a new camera."""
    return CameraService.create_camera(data.model_dump())


@router.get("/{camera_id}")
async def get_camera(
    camera_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get camera details and status."""
    return CameraService.get_camera(camera_id)


@router.put("/{camera_id}", response_model=CameraResponse)
async def update_camera(
    camera_id: str,
    data: CameraUpdate,
    current_user: dict = Depends(get_current_user),
):
    """Update camera configuration."""
    return CameraService.update_camera(camera_id, data.model_dump(exclude_unset=True))


@router.delete("/{camera_id}")
async def delete_camera(
    camera_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Remove a camera."""
    CameraService.delete_camera(camera_id)
    return {"message": f"Camera {camera_id} deleted"}


@router.post("/{camera_id}/start")
async def start_camera(
    camera_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Start AI processing for a camera."""
    return CameraService.start_camera(camera_id)


@router.post("/{camera_id}/stop")
async def stop_camera(
    camera_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Stop AI processing for a camera."""
    return CameraService.stop_camera(camera_id)


# ── MJPEG Streaming (no auth required for <img> tags) ─────────

@router.get("/{camera_id}/feed")
async def camera_feed(camera_id: str):
    """
    HTTP MJPEG feed / Snapshot endpoint.
    Returns the latest pre-encoded JPEG frame.
    """
    from fastapi.responses import Response

    worker = camera_manager._workers.get(camera_id)
    if not worker or not worker.pipeline:
        return Response(content=b"", status_code=204)

    # Use the pre-encoded JPEG from the worker thread
    jpeg = worker.pipeline._last_jpeg
    
    # Fallback if no frame processed yet
    if jpeg is None:
        return Response(content=b"", status_code=204)

    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ── WebSocket Stream ──────────────────────────────────────────

@router.websocket("/{camera_id}/stream")
async def camera_stream(
    websocket: WebSocket,
    camera_id: str,
):
    """
    WebSocket endpoint for live camera stream.
    Reads pre-encoded JPEG bytes from the pipeline (encoded in the worker thread).
    Zero CPU work happens here — just read cached bytes and send.
    """
    await websocket.accept()
    logger.info(f"Stream WebSocket connected for camera: {camera_id}")

    last_jpeg_id = None  # Track which JPEG blob we last sent

    try:
        while True:
            worker = camera_manager._workers.get(camera_id)
            if not worker or not worker.pipeline:
                await asyncio.sleep(0.1)
                continue

            # Read pre-encoded JPEG from pipeline (encoded in worker thread)
            jpeg_bytes = worker.pipeline._last_jpeg
            if jpeg_bytes is None:
                await asyncio.sleep(0.05)
                continue

            # Only send if the JPEG data has actually changed
            current_id = id(jpeg_bytes)
            if current_id != last_jpeg_id:
                last_jpeg_id = current_id
                try:
                    await websocket.send_bytes(jpeg_bytes)
                except Exception:
                    break
            
            # ~30fps push rate (33ms) — minimal latency since no encoding happens here
            await asyncio.sleep(0.033)

    except WebSocketDisconnect:
        logger.info(f"Stream WebSocket disconnected for camera: {camera_id}")
    except Exception as e:
        logger.error(f"Stream WebSocket error for camera {camera_id}: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
