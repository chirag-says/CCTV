"""
Video Analysis API Routes.

Endpoints for uploading videos and retrieving AI analysis results.
"""

import asyncio
import logging
import os
import shutil
from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse
from typing import Optional

from app.config import settings
from app.core.security import get_current_user
from app.services.video_analysis_service import video_analysis_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/video-analysis", tags=["Video Analysis"])

# Ensure upload directory exists
os.makedirs(settings.VIDEO_UPLOAD_DIR, exist_ok=True)


def _get_allowed_extensions() -> set:
    """Get set of allowed video file extensions."""
    return {
        ext.strip().lower()
        for ext in settings.ALLOWED_VIDEO_EXTENSIONS.split(",")
    }


@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """
    Upload a video file and start AI analysis.

    Accepts video files (mp4, avi, mkv, mov, wmv, flv, webm).
    Returns a job_id to track analysis progress.
    """
    # Validate file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    allowed = _get_allowed_extensions()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '.{ext}'. Allowed: {', '.join(sorted(allowed))}",
        )

    # Validate file size (read content-length header if available)
    max_size = settings.MAX_VIDEO_SIZE_MB * 1024 * 1024

    # Save file to disk
    job_id = str(uuid4())
    safe_filename = f"{job_id}.{ext}"
    video_path = os.path.join(settings.VIDEO_UPLOAD_DIR, safe_filename)

    try:
        os.makedirs(settings.VIDEO_UPLOAD_DIR, exist_ok=True)
        total_written = 0

        with open(video_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)  # Read 1MB chunks
                if not chunk:
                    break
                total_written += len(chunk)
                if total_written > max_size:
                    # Clean up and reject
                    f.close()
                    os.remove(video_path)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size: {settings.MAX_VIDEO_SIZE_MB}MB",
                    )
                f.write(chunk)

        logger.info(
            f"Video uploaded: {file.filename} ({total_written / 1024 / 1024:.1f}MB) → {video_path}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save uploaded video: {e}")
        if os.path.exists(video_path):
            os.remove(video_path)
        raise HTTPException(status_code=500, detail="Failed to save video file")

    # Start analysis
    try:
        analysis_job_id = video_analysis_manager.start_analysis(
            video_path=video_path,
            filename=file.filename,
        )
        return {
            "job_id": analysis_job_id,
            "filename": file.filename,
            "size_mb": round(total_written / 1024 / 1024, 2),
            "status": "processing",
            "message": "Video uploaded successfully. Analysis started.",
        }
    except RuntimeError as e:
        # Max concurrent analyses reached — clean up file
        if os.path.exists(video_path):
            os.remove(video_path)
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start video analysis: {e}")
        if os.path.exists(video_path):
            os.remove(video_path)
        raise HTTPException(status_code=500, detail="Failed to start video analysis")


@router.get("/{job_id}/status")
async def get_analysis_status(
    job_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get the current status and progress of a video analysis job."""
    status = video_analysis_manager.get_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Analysis job not found")
    return status


@router.get("/{job_id}/results")
async def get_analysis_results(
    job_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Get the full results of a completed video analysis.
    Includes all detected persons, vehicles, alerts, and event timeline.
    """
    results = video_analysis_manager.get_results(job_id)
    if not results:
        raise HTTPException(status_code=404, detail="Analysis job not found")

    if results["status"] == "processing":
        return JSONResponse(
            status_code=202,
            content={
                "message": "Analysis still in progress",
                "progress": results["progress"],
                "status": results["status"],
            },
        )

    return results


@router.get("/history")
async def get_analysis_history(
    current_user: dict = Depends(get_current_user),
):
    """Get a list of all video analysis jobs (past and current)."""
    return {
        "analyses": video_analysis_manager.get_all_jobs(),
        "total": len(video_analysis_manager.get_all_jobs()),
    }


@router.delete("/{job_id}")
async def delete_analysis(
    job_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Delete a video analysis job and its associated files."""
    deleted = video_analysis_manager.delete_job(job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Analysis job not found")
    return {"message": f"Analysis {job_id} deleted successfully"}


# ── WebSocket for Real-Time Progress ─────────────────────────

@router.websocket("/{job_id}/stream")
async def analysis_progress_stream(
    websocket: WebSocket,
    job_id: str,
):
    """
    WebSocket endpoint for real-time analysis progress updates.
    Sends JSON messages with progress percentage and status.
    """
    await websocket.accept()
    logger.info(f"Analysis progress WebSocket connected: {job_id}")

    try:
        while True:
            status = video_analysis_manager.get_status(job_id)
            if not status:
                await websocket.send_json({
                    "error": "Job not found",
                    "job_id": job_id,
                })
                break

            await websocket.send_json({
                "job_id": job_id,
                "status": status["status"],
                "progress": status["progress"],
                "video_metadata": status.get("video_metadata", {}),
            })

            # Stop streaming when job is done
            if status["status"] in ("completed", "failed"):
                # Send final summary
                results = video_analysis_manager.get_results(job_id)
                if results:
                    await websocket.send_json({
                        "job_id": job_id,
                        "status": status["status"],
                        "progress": 100,
                        "summary": results.get("summary", {}),
                        "completed": True,
                    })
                break

            await asyncio.sleep(0.5)  # Update every 500ms

    except WebSocketDisconnect:
        logger.info(f"Analysis progress WebSocket disconnected: {job_id}")
    except Exception as e:
        logger.error(f"Analysis progress WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
