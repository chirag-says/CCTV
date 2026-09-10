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


@router.get("/debug/recognition-state")
async def debug_recognition_state(
    current_user: dict = Depends(get_current_user),
):
    """
    Debug endpoint: shows how many face encodings are loaded in each camera's recognizer.
    Use this to verify that registered faces are actually loaded for recognition.
    """
    from app.services.person_service import PersonService

    db_encodings = PersonService.get_all_encodings()
    pipeline_states = {}

    for camera_id in camera_manager.get_active_camera_ids():
        worker = camera_manager._workers.get(camera_id)
        if worker and worker.pipeline:
            recognizer = worker.pipeline.recognizer
            pipeline_states[camera_id] = {
                "known_count": recognizer.known_count,
                "known_persons": recognizer.known_persons,
                "tolerance": recognizer.tolerance,
                "pipeline_mode": worker.pipeline.pipeline_mode,
            }

    return {
        "db_encoding_count": len(db_encodings),
        "db_persons": list(set(e["person_name"] for e in db_encodings)),
        "cached_encodings": len(camera_manager._cached_encodings) if camera_manager._cached_encodings else 0,
        "active_pipelines": pipeline_states,
        "issue_detected": any(
            p["known_count"] == 0 for p in pipeline_states.values()
        ) if pipeline_states else None,
    }


@router.post("/refresh-encodings")
async def refresh_encodings(
    current_user: dict = Depends(get_current_user),
):
    """
    Force reload face encodings from DB into all active camera pipelines.
    Use this if recognition seems broken after adding new faces.
    """
    count = camera_manager.refresh_encodings()
    return {
        "message": f"Refreshed {count} encodings into all active pipelines",
        "encoding_count": count,
        "active_cameras": len(camera_manager.get_active_camera_ids()),
    }


@router.get("/active-persons")
async def get_active_persons(
    current_user: dict = Depends(get_current_user),
):
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
    """List all registered cameras with status, including active browser cameras."""
    db_cameras = CameraService.list_cameras(is_active=is_active)

    # Also include active browser camera sessions (skip if already in DB results)
    existing_ids = {c.get("id") or getattr(c, "id", None) for c in db_cameras}
    from app.routes.browser_camera import _browser_sessions
    for sid, session in _browser_sessions.items():
        if session.is_active and sid not in existing_ids:
            db_cameras.append({
                "id": sid,
                "name": session.label,
                "location": "Remote (Browser)",
                "stream_url": f"browser:{sid}",
                "status": "online",
                "is_active": True,
                "type": "browser",
                "group": "Remote (Browser)",
                "fps": session.status.get("fps", 0),
                "frame_count": session.frame_count,
            })

    return db_cameras


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
    
    # Also clean up from browser sessions if it's a remote camera
    from app.routes.browser_camera import _browser_sessions
    if camera_id in _browser_sessions:
        _browser_sessions[camera_id].is_active = False
        _browser_sessions.pop(camera_id, None)
        
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


# ── Live view: WebRTC via go2rtc ──────────────────────────────

@router.get("/{camera_id}/webrtc")
async def camera_webrtc(
    camera_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Where the browser signals to play this camera over WebRTC.

    Video does not pass through this process. go2rtc pulls the camera's RTSP
    (or, in development, loops a file through ffmpeg) and negotiates WebRTC
    with the browser directly; all this endpoint does is say where.

    `available` false is a normal answer, not an error: a webcam-index camera
    is already held by the analysis pipeline and a browser-pushed session has
    no source to relay. The monitor page falls back to MJPEG on it.
    """
    from app.services.go2rtc import go2rtc

    CameraService.get_camera(camera_id)  # 404s for an unknown camera

    status = go2rtc.status()
    available = bool(status["enabled"] and go2rtc.has_stream(camera_id))

    return {
        "camera_id": camera_id,
        "available": available,
        "signalling_url": go2rtc.signalling_url(camera_id) if available else None,
        "metadata_url": f"/api/cameras/{camera_id}/metadata",
        "relay": {
            "running": status["running"],
            "restarts": status["restarts"],
            "last_error": status["last_error"],
        },
    }


# ── Detection metadata (no pixels on this channel) ────────────

@router.websocket("/{camera_id}/metadata")
async def camera_metadata(
    websocket: WebSocket,
    camera_id: str,
):
    """One JSON message per analysed frame: timestamp, track ids, normalised
    boxes, labels, identities. Person and face only.

    This socket carries no image data of any kind. It is the companion to the
    WebRTC video above -- the browser draws these boxes on a canvas over the
    video it is already receiving from go2rtc.

    Requires a JWT via query parameter, exactly like the stream socket:
      ws://host/api/cameras/{id}/metadata?token=<jwt>
    """
    from app.core.security import require_ws_token

    if await require_ws_token(websocket) is None:
        return  # already closed with 4001

    await websocket.accept()
    await ws_manager.connect_metadata(camera_id, websocket)

    try:
        # Nothing is expected FROM the client except keep-alives; this read is
        # how a disconnect is noticed. Messages are pushed from the pipeline
        # through ws_manager, not produced here.
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info(f"Metadata WebSocket disconnected for camera: {camera_id}")
    except Exception as e:
        logger.debug(f"Metadata WebSocket error for camera {camera_id}: {e}")
    finally:
        await ws_manager.disconnect_metadata(camera_id, websocket)
        try:
            await websocket.close()
        except Exception:
            pass


# ── MJPEG Streaming (snapshot endpoint and fallback only) ─────

@router.get("/{camera_id}/feed")
async def camera_feed(
    camera_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    HTTP snapshot endpoint — returns the latest pre-encoded JPEG frame.

    Requires authentication. This previously had no auth at all and sent
    `Access-Control-Allow-Origin: *`, so any website could embed a live
    annotated camera feed from this deployment.
    """
    from fastapi.responses import Response

    # Count this as a viewer BEFORE the early returns: a client polling a
    # camera that is briefly not LIVE is still watching, and if we only
    # registered on the success path the encoder would never restart.
    ws_manager.note_snapshot_request(camera_id)

    worker = camera_manager._workers.get(camera_id)
    if not worker or not worker.pipeline:
        return Response(content=b"", status_code=204)

    # Never return stale JPEG — return 204 if stream is not LIVE
    from app.vision.stream_health import StreamState
    if worker.stream_health.state != StreamState.LIVE:
        return Response(content=b"", status_code=204)

    # Pre-encoded bytes if an encoder thread happens to be running for a
    # fallback viewer; otherwise encode the last frame right here.
    #
    # Since Phase 3 the encoder thread no longer starts with the camera, so
    # on the default path there is nothing to read. A snapshot is a single
    # frame on request -- one 8.9 ms encode on the request thread is the whole
    # cost, against a thread per camera burning that 30 times a second.
    jpeg = worker.pipeline._last_jpeg
    if jpeg is None:
        frame = worker.pipeline._last_frame
        if frame is None:
            return Response(content=b"", status_code=204)
        from app.utils.image_utils import frame_to_jpeg_bytes
        jpeg = frame_to_jpeg_bytes(frame, quality=80)

    if not jpeg:
        return Response(content=b"", status_code=204)

    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


# ── WebSocket Stream ──────────────────────────────────────────

@router.websocket("/{camera_id}/stream")
async def camera_stream(
    websocket: WebSocket,
    camera_id: str,
):
    """
    MJPEG-over-WebSocket fallback for live video.

    NO LONGER THE DEFAULT. Since Phase 3 video reaches the browser from
    go2rtc over WebRTC and this process sends metadata only; this endpoint
    exists for cameras go2rtc cannot relay (a webcam index, a browser-pushed
    session) and for a viewer who asks for it explicitly with ?fallback=1.

    Connecting here starts the camera's JPEG encoder thread if it is not
    already running, and it stops again when the last viewer leaves.

    Requires a JWT via query parameter:
      ws://host/api/cameras/{id}/stream?token=<jwt>&fallback=1
    """
    from app.core.security import require_ws_token

    if await require_ws_token(websocket) is None:
        return  # already closed with 4001

    await websocket.accept()
    # Register as a viewer of this camera. Until this existed, nothing ever
    # populated ws_manager's stream table, so the worker had no way to know
    # whether anyone was looking -- and encoded JPEG for all of them anyway.
    await ws_manager.connect_stream(camera_id, websocket)

    # Start the encoder on demand. `is_watched` already gates whether frames
    # are encoded; this gates whether the THREAD exists at all, which is the
    # part Phase 3 removes from the default path.
    _worker = camera_manager._workers.get(camera_id)
    if _worker:
        _worker.ensure_jpeg_encoder()

    logger.info(f"Stream WebSocket (MJPEG fallback) connected for camera: {camera_id}")

    last_jpeg_id = None  # Track which JPEG blob we last sent

    try:
        while True:
            jpeg_bytes = None

            # Priority 1: AI-processed frame from pipeline (has bounding boxes)
            worker = camera_manager._workers.get(camera_id)
            if worker and worker.pipeline:
                # Check stream health — send JSON stale notification instead of JPEG
                from app.vision.stream_health import StreamState
                health = worker.stream_health
                if health.state != StreamState.LIVE:
                    try:
                        import json
                        stale_msg = json.dumps({
                            "type": "stream_stale",
                            "stream_state": health.state.value,
                            "stale_since": health.stale_since.isoformat() if health.stale_since else None,
                            "reconnect_attempts": health.reconnect_attempts,
                            "last_frame_age": round(health.last_frame_age, 1),
                        })
                        await websocket.send_text(stale_msg)
                    except Exception:
                        break
                    await asyncio.sleep(1.0)  # Slower poll during stale
                    continue

                if worker.pipeline._last_jpeg:
                    jpeg_bytes = worker.pipeline._last_jpeg
            else:
                # Priority 2: Raw frame from browser camera session
                from app.routes.browser_camera import _browser_sessions
                browser_session = _browser_sessions.get(camera_id)
                if browser_session and browser_session.is_active:
                    jpeg_bytes = browser_session.last_jpeg
                else:
                    await asyncio.sleep(0.1)
                    continue

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
            
            # Fast polling for low-latency relay
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        logger.info(f"Stream WebSocket disconnected for camera: {camera_id}")
    except Exception as e:
        logger.error(f"Stream WebSocket error for camera {camera_id}: {e}")
    finally:
        await ws_manager.disconnect_stream(camera_id, websocket)
        try:
            await websocket.close()
        except Exception:
            pass

