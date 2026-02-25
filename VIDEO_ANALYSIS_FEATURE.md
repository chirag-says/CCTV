# 🎬 Video Upload Analysis Feature — Implementation Plan

## Feature Overview

Allow users to **upload a pre-recorded video file** and run the same AI-powered analysis pipeline that currently operates on live webcam/camera feeds. The system will process the uploaded video frame-by-frame and extract all intelligence:

- **Face Recognition** — Identify known persons, detect unknown faces
- **Vehicle / License Plate Detection (ANPR)** — Read vehicle number plates
- **Person Tracking** — Entry/exit tracking with timestamps
- **Hazard Detection** — Detect weapons, dangerous objects
- **Crowd / Gathering Detection** — Identify groups of people
- **Loitering Detection** — Identify people staying in one area
- **Traffic Safety** — Vehicle-pedestrian proximity alerts
- **Attribute Recognition** — Gender, clothing color

The results are presented in a comprehensive **analysis report** with a timeline of events, detected persons, vehicles, and security alerts.

---

## Architecture Design

```
┌──────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Next.js)                        │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              VIDEO ANALYSIS PAGE (/video-analysis)        │   │
│  │                                                           │   │
│  │  ┌─────────────┐  ┌──────────────────┐  ┌─────────────┐ │   │
│  │  │ Upload Zone  │  │  Analysis Status │  │  Results     │ │   │
│  │  │ (Drag+Drop)  │  │  (Progress Bar)  │  │  Dashboard   │ │   │
│  │  └─────────────┘  └──────────────────┘  └─────────────┘ │   │
│  │                                                           │   │
│  │  ┌─────────────────────────────────────────────────────┐ │   │
│  │  │              EVENT TIMELINE                          │ │   │
│  │  │  Detected Persons │ Vehicles │ Alerts │ Timeline     │ │   │
│  │  └─────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┬───────┘
                                                           │
                                                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                        BACKEND (FastAPI)                         │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │               VIDEO ANALYSIS MODULE                       │   │
│  │                                                           │   │
│  │  POST /api/video-analysis/upload   → Accept video file    │   │
│  │  GET  /api/video-analysis/{id}/status → Analysis progress │   │
│  │  GET  /api/video-analysis/{id}/results → Full results     │   │
│  │  GET  /api/video-analysis/history  → Past analyses        │   │
│  │  WS   /api/video-analysis/{id}/stream → Live progress     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │               VIDEO PIPELINE PROCESSOR                    │   │
│  │                                                           │   │
│  │  Reuses existing Vision Pipeline components:              │   │
│  │  ├── FaceDetector          (face detection)               │   │
│  │  ├── FaceRecognizer        (face matching)                │   │
│  │  ├── PersonTracker         (entry/exit logic)             │   │
│  │  ├── HazardDetector        (weapon detection)             │   │
│  │  ├── PlateRecognizer       (ANPR)                         │   │
│  │  ├── TrafficMonitor        (vehicle-pedestrian safety)    │   │
│  │  ├── CrowdDetector         (crowd/gathering)              │   │
│  │  ├── LoiteringDetector     (idle person detection)        │   │
│  │  └── AttributeRecognizer   (gender + clothing)            │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Implementation Steps

### Phase 1: Backend — Video Analysis Engine

#### Step 1.1: Create Video Analysis Service (`backend/app/services/video_analysis_service.py`)

- **VideoAnalysisJob** class that manages the lifecycle of a single video analysis
- Accepts a video file path and processes it frame-by-frame
- Reuses existing `FaceDetector`, `FaceRecognizer`, `PersonTracker`, `HazardDetector`, `PlateRecognizer`, `TrafficMonitor`, `CrowdDetector`, `LoiteringDetector`, `AttributeRecognizer`
- Tracks analysis progress (0-100%)
- Collects all events into an in-memory results store
- Runs in a background thread to avoid blocking the API

**Key Data Structures:**
```python
class VideoAnalysisResult:
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: float  # 0.0 to 100.0
    video_metadata: dict  # fps, duration, resolution, total_frames
    detected_persons: List[dict]  # All recognized + unknown faces
    detected_vehicles: List[dict]  # All ANPR detections
    security_alerts: List[dict]   # Hazard, crowd, loitering alerts
    traffic_alerts: List[dict]    # Vehicle-pedestrian proximity
    events_timeline: List[dict]   # Chronological event log
    summary: dict                 # Aggregate statistics
    annotated_frames: List[str]   # Key frame snapshot URLs
```

#### Step 1.2: Create Video Analysis Routes (`backend/app/routes/video_analysis.py`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/video-analysis/upload` | POST | Upload video file, start analysis |
| `/api/video-analysis/{job_id}/status` | GET | Get analysis progress |
| `/api/video-analysis/{job_id}/results` | GET | Get full analysis results |
| `/api/video-analysis/history` | GET | List all past analyses |
| `/api/video-analysis/{job_id}` | DELETE | Delete analysis and files |
| `/api/video-analysis/{job_id}/stream` | WS | Real-time progress updates |

#### Step 1.3: Update Configuration (`backend/app/config.py`)

Add new settings:
```python
# ── Video Analysis ────────────────────────────────────────
VIDEO_UPLOAD_DIR: str = "./uploads/videos"
MAX_VIDEO_SIZE_MB: int = 500
ALLOWED_VIDEO_EXTENSIONS: str = "mp4,avi,mkv,mov,wmv,flv,webm"
VIDEO_ANALYSIS_FRAME_SKIP: int = 5  # Process every 5th frame for speed
VIDEO_ANALYSIS_MAX_CONCURRENT: int = 2  # Max simultaneous analyses
```

#### Step 1.4: Register Routes in `main.py`

Add the video analysis router to the FastAPI application.

### Phase 2: Frontend — Video Analysis Page

#### Step 2.1: Create Video Analysis Page (`frontend/src/app/video-analysis/page.js`)

Full-featured page with:
- **Drag & drop upload zone** with file validation
- **Analysis progress** with real-time WebSocket updates
- **Results dashboard** with tabs:
  - 📊 Summary — Key metrics at a glance
  - 👤 Persons — All detected faces (known + unknown) with timestamps
  - 🚗 Vehicles — Detected license plates with frame snapshots
  - 🚨 Alerts — Security alerts (hazards, crowds, loitering)
  - 📋 Timeline — Chronological event log
- **Key frame viewer** — Annotated frames from significant events

#### Step 2.2: Update API Client (`frontend/src/lib/api.js`)

Add methods:
```javascript
uploadVideo(file, onProgress)        // Upload with progress callback
getAnalysisStatus(jobId)             // Poll for status
getAnalysisResults(jobId)            // Get full results
getAnalysisHistory()                 // List past analyses
deleteAnalysis(jobId)                // Delete analysis
connectAnalysisStream(jobId, onMsg)  // WebSocket for live updates
```

#### Step 2.3: Update Sidebar Navigation

Add "Video Analysis" link with a video icon in the sidebar.

---

## File Changes Summary

### New Files
| File | Purpose |
|------|---------|
| `backend/app/services/video_analysis_service.py` | Core analysis engine |
| `backend/app/routes/video_analysis.py` | API endpoints |
| `frontend/src/app/video-analysis/page.js` | Frontend page |

### Modified Files
| File | Changes |
|------|---------|
| `backend/app/config.py` | Add video analysis settings |
| `backend/app/main.py` | Register video analysis routes |
| `frontend/src/lib/api.js` | Add video analysis API methods |
| `frontend/src/components/Sidebar.js` | Add navigation link |

---

## Technical Considerations

### Video Processing Strategy
- Process frames in a **background thread** (same pattern as `CameraWorker`)
- Use **frame skipping** (configurable, default every 5th frame) for performance
- Save **key annotated frames** at significant events (face detected, plate read, alert)
- Track **video timestamps** (not wall-clock time) for accurate event timing

### Memory Management
- Process frames one at a time — no buffering of entire video
- Release OpenCV capture after processing
- Clean up temporary files after analysis

### Pipeline Reuse
- Create instances of all existing vision modules (`FaceDetector`, `FaceRecognizer`, etc.)
- Load face encodings from database (same as camera pipeline)
- All detection thresholds and settings apply identically

### Error Handling
- File validation (size, format, corruption check)
- Graceful failure with status updates
- Timeout for extremely long videos

---

## Implementation Order

1. ✅ Create this implementation plan (this file)
2. ✅ Backend: Add config settings
3. ✅ Backend: Create `video_analysis_service.py`
4. ✅ Backend: Create `video_analysis.py` routes
5. ✅ Backend: Register routes in `main.py`
6. ✅ Frontend: Add API methods to `api.js`
7. ✅ Frontend: Create Video Analysis page
8. ✅ Frontend: Update Sidebar navigation
9. 🧪 Test end-to-end flow
